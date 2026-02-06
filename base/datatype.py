import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, TypeAlias

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from sophuspy import SE3

from base.interpolate import get_time_series, interpolate_vector3, slerp_rotation
from base.rtab import RTABData


@dataclass
class Pose:
    rot: Rotation
    p: NDArray
    t_us: int = 0

    def copy(self):
        return Pose(self.rot, self.p.copy(), self.t_us)

    @staticmethod
    def identity():
        return Pose(Rotation.identity(), np.zeros(3))

    @staticmethod
    def from_rotation(rot: Rotation):
        return Pose(rot, np.zeros(3))

    @staticmethod
    def from_transform(trans: NDArray):
        return Pose(Rotation.identity(), trans)

    def compose(self, other: Self):
        return Pose(self.rot * other.rot, self.rot.apply(other.p) + self.p)

    def compose_self(self, other: Self):
        self.p = self.rot.apply(other.p) + self.p
        self.rot = self.rot * other.rot

    def compose_trans_self(self, trans: NDArray):
        self.p += self.rot.apply(trans)

    def inverse(self):
        return Pose(self.rot.inv(), -self.rot.inv().apply(self.p))

    def between(self, to: Self):
        return self.inverse().compose(to)

    def log(self):
        return SE3(self.rot.as_matrix(), self.p).log()

    def get_yaw_pose(self):
        yaw = self.rot.as_euler("ZXY")[0]
        return Pose(Rotation.from_rotvec([0, 0, yaw]), np.zeros(3))


Frame: TypeAlias = Literal["local", "global"]


@dataclass
class PosesData:
    t_us: NDArray
    rots: Rotation
    ps: NDArray

    def __len__(self):
        return len(self.t_us)

    def __getitem__(self, index: int | slice):
        return PosesData(self.t_us[index], self.rots[index], self.ps[index])

    def __iter__(self):
        for i in range(len(self.t_us)):
            yield Pose(self.rots[i], self.ps[i], self.t_us[i])

    @staticmethod
    def from_list(pose_list: list[Pose]):
        rotations = []
        translates = []
        t_us = []
        for pose in pose_list:
            rotations.append(pose.rot.as_matrix())
            translates.append(pose.p)
            t_us.append(pose.t_us)

        t_us = np.array(t_us)
        rots = Rotation.from_matrix(rotations)
        ps = np.array(translates)
        return PosesData(t_us, rots, ps)

    def get_pose(self, index: int = 0):
        return Pose(self.rots[index], self.ps[index], self.t_us[index])

    def get_time_range(self, time_range: tuple[float | None, float | None]):
        ts, te = self.t_us[0], self.t_us[-1]
        if time_range[0] is not None:
            ts = time_range[0] * 1e6 + self.t_us[0]
        if time_range[1] is not None:
            te = time_range[1] * 1e6 + self.t_us[0]
        return self.get_time_region((ts, te))

    def get_time_region(self, t_region: tuple[int, int]) -> "PosesData":
        ts, te = t_region
        mask = (self.t_us >= ts) & (self.t_us <= te)
        return PosesData(self.t_us[mask], self.rots[mask], self.ps[mask])

    def interpolate(self, t_new_us: NDArray, bounds_error: bool = False) -> "PosesData":
        rots = slerp_rotation(self.rots, self.t_us, t_new_us)
        trans = interpolate_vector3(self.ps, self.t_us, t_new_us, bounds_error)
        return PosesData(t_new_us, rots, trans)

    def transform_local(self, tf: Pose):
        # R = R * R_loc
        # t = t + R * t_loc
        self.ps = self.ps + self.rots.apply(tf.p)
        self.rots = self.rots * tf.rot

    def transform_global(self, tf: Pose):
        # R = R_glo * R
        # t = t_glo + R_glo * t
        self.rots = tf.rot * self.rots
        self.ps = tf.p + tf.rot.apply(self.ps)

    def get_between(self, s: int, e: int) -> Pose:
        s_pose = self.get_pose(s)
        e_pose = self.get_pose(e)
        rel_pose = s_pose.between(e_pose)
        return rel_pose

    def reset_start(self):
        """重置出发点"""
        self.ps -= self.ps[0]

    @property
    def rate(self):
        return float(1e6 / np.mean(np.diff(self.t_us)))

    @property
    def length_meter(self):
        """
        获取轨迹长度 meter
        """
        disps = np.diff(self.ps, axis=0)
        disp_lens = np.linalg.norm(disps, axis=1)
        return np.sum(disp_lens)

    def length_meter_fix_rate(self, rate: int = 1):
        """
        设定固定频率 获取轨迹长度 meter
        """
        t_new_us = get_time_series([self.t_us], rate=rate)
        return self.interpolate(t_new_us).length_meter


@dataclass
class ImuData:
    t_us: NDArray
    gyro: NDArray
    acce: NDArray
    ahrs: Rotation
    magn: NDArray

    frame: Frame = "local"
    # 补充内容
    # acce_bias: NDArray = np.zeros(0)
    # gyro_bias: NDArray = np.zeros(0)

    def __getitem__(self, idx):
        return ImuData(
            self.t_us[idx],
            self.gyro[idx],
            self.acce[idx],
            self.ahrs[idx],
            self.magn[idx],
            self.frame,
        )

    def __len__(self):
        return self.t_us.__len__()

    @staticmethod
    def from_raw(raw: NDArray) -> "ImuData":
        assert raw.shape[1] >= 12, f"Invalid raw data shape: {raw.shape}"
        gyro = raw[:, 1:4]
        acce = raw[:, 4:7]
        ahrs = Rotation.from_quat(raw[:, 7:11], scalar_first=True)
        t_us = raw[:, 0] + raw[:, 11][0] + -raw[:, 0][0]

        if raw.shape[1] >= 15:
            magn = raw[:, 12:15]
        else:
            magn = np.zeros_like(gyro)
            # raise ValueError("Invalid raw data shape: missing magnetometer data")
        return ImuData(t_us, gyro, acce, ahrs, magn)

    @staticmethod
    def from_csv(path: Path) -> "ImuData":
        raw = pd.read_csv(path).dropna().to_numpy()
        return ImuData.from_raw(raw)

    def interpolate(
        self,
        t_new_us: NDArray,
        bounds_error: bool = False,
    ) -> "ImuData":
        acce = interpolate_vector3(self.acce, self.t_us, t_new_us, bounds_error)
        gyro = interpolate_vector3(self.gyro, self.t_us, t_new_us, bounds_error)
        ahrs = slerp_rotation(self.ahrs, self.t_us, t_new_us)
        # 磁场
        magn = interpolate_vector3(self.magn, self.t_us, t_new_us, bounds_error)
        return ImuData(t_new_us, gyro, acce, ahrs, magn)

    def transform(self, rots: Rotation | None = None):
        if rots is None:
            rots = self.ahrs
        acce = rots.apply(self.acce)
        gyro = rots.apply(self.gyro)
        magn = rots.apply(self.magn)
        ahrs = rots * self.ahrs
        return ImuData(self.t_us, gyro, acce, ahrs, magn, frame="global")

    def get_time_range(self, time_range: tuple[float | None, float | None]):
        ts, te = self.t_us[0], self.t_us[-1]
        if time_range[0]:
            ts = time_range[0] * 1e6 + self.t_us[0]
        if time_range[1]:
            te = time_range[1] * 1e6 + self.t_us[0]
        m = (self.t_us >= ts) & (self.t_us <= te)
        return ImuData(
            self.t_us[m],
            self.gyro[m],
            self.acce[m],
            self.ahrs[m],
            self.magn[m],
            self.frame,
        )

    def to_poses(self) -> PosesData:
        ps = np.zeros((len(self.t_us), 3))
        return PosesData(self.t_us, self.ahrs, ps)
    

    def calibrate_with(self, config_path: Path):
        imu_calib = ImuCalibration(config_path)

        self.acce = imu_calib.ra.apply(self.acce) - imu_calib.ba
        self.gyro = imu_calib.rg.apply(self.gyro) - imu_calib.bg
        


class ImuCalibration:
    def __init__(self, config_path: Path):
        self.ra = Rotation.identity()
        self.rg = Rotation.identity()
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if "Calibration" in config:
                calib = config["Calibration"]
                ra_arr = np.array(calib["ra"]).reshape((3, 3))
                rg_arr = np.array(calib["rg"]).reshape((3, 3))
                self.ra = Rotation.from_matrix(ra_arr)
                self.rg = Rotation.from_matrix(rg_arr)
                self.ba = np.array(calib["ba"])
                self.bg = np.array(calib["bg"])

    def calibrate(self, imu_data: ImuData):
        imu_data.acce = self.ra.apply(imu_data.acce) - self.ba
        imu_data.gyro = self.rg.apply(imu_data.gyro) - self.bg
        return imu_data

    def __repr__(self):
        return f"ImuCalibration(Ra={self.ra}, Rg={self.rg}, Ba={self.ba}, Bg={self.bg})"


class GroundTruthData(PosesData):
    @staticmethod
    def from_raw(raw: NDArray):
        t_us = raw[:, 0]
        trans = raw[:, 1:4]
        quats = raw[:, 4:8]
        # qwxyz
        rots = Rotation.from_quat(quats, scalar_first=True)
        return GroundTruthData(t_us, rots, trans)

    @staticmethod
    def from_csv(path: Path):
        raw = pd.read_csv(path).to_numpy()
        return GroundTruthData.from_raw(raw)


class CameraColumn:
    """
    #timestamp [us],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],p_CS_C_x [m],p_CS_C_y [m],p_CS_C_z [m],q_CS_w [],q_CS_x [],q_CS_y [],q_CS_z [],t_system [us]
    """

    t = ["#timestamp [us]"]
    ps = ["p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]"]
    qs = ["q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"]
    pc = ["p_CS_C_x [m]", "p_CS_C_y [m]", "p_CS_C_z [m]"]
    qc = ["q_CS_w []", "q_CS_x []", "q_CS_y []", "q_CS_z []"]
    t_sys = ["t_system [us]"]

    all = t + ps + qs + pc + qc + t_sys


class CameraData(PosesData):
    @staticmethod
    def from_csv(path: Path):
        df = (
            pd.read_csv(path)
            .sort_values(by=CameraColumn.t)
            .drop_duplicates(CameraColumn.t)
        )
        t_sensor_us = df[CameraColumn.t].to_numpy().flatten()
        t_us = df[CameraColumn.t_sys].to_numpy().flatten()
        trans = df[CameraColumn.ps].to_numpy()
        rots = df[CameraColumn.qs].to_numpy()
        rots = Rotation.from_quat(rots, scalar_first=True)

        t_us = t_sensor_us - t_sensor_us[0] + t_us[0]
        return CameraData(t_us, rots, trans)


class FusionData(PosesData):
    """
    t_us: 此时间为系统时间，非UTC时间戳
    """

    @staticmethod
    def from_raw(raw: NDArray):
        t_us = raw[:, 0]
        trans = raw[:, 1:4]
        quats = raw[:, 4:8]
        # cam_trans = raw[:, 8:11]
        # cam_quats = raw[:, 11:15]
        rots = Rotation.from_quat(quats, scalar_first=True)
        return FusionData(t_us, rots, trans)

    @staticmethod
    def from_csv(path: Path):
        raw = pd.read_csv(path).drop_duplicates(CameraColumn.t).to_numpy()
        return FusionData.from_raw(raw)


@dataclass
class CalibrationData:
    tf_sg_local: Pose
    tf_sg_global: Pose

    @staticmethod
    def from_json(path: Path):
        with open(path, "r") as f:
            data = json.load(f)
            if not isinstance(data, list) or len(data) != 1:
                raise ValueError("Invalid JSON format")
            data = data[0]

            rot_local = np.array(data["rot_sensor_gt"])
            trans_local = np.array(data["trans_sensor_gt"]).flatten()
            tf_sg_local = Pose(Rotation.from_matrix(rot_local), trans_local)

            rot_global = np.array(data["rot_ref_sensor_gt"])
            trans_global = np.array(data["trans_ref_sensor_gt"]).flatten()
            tf_sg_global = Pose(Rotation.from_matrix(rot_global), trans_global)

            print(f"Load Calib File: {path}")

            return CalibrationData(tf_sg_local, tf_sg_global)


@dataclass
class DataCheck:
    t_gi_us: int

    @staticmethod
    def from_json(path: Path):
        with open(path, "r") as f:
            data = json.load(f)
            assert "check_time_diff" in data
            check_time_diff = data["check_time_diff"]
            assert "time_diff_21_us" in check_time_diff
            t_gi_us = check_time_diff["time_diff_21_us"]
            return DataCheck(t_gi_us)


class UnitData:
    imu_data: ImuData
    gt_data: PosesData
    opt_data: PosesData

    has_opt: bool
    using_ext: bool = False

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.name = self.base_dir.name

        # 设备名称
        spl = self.name.split("_")
        device_name = spl[2] if len(spl) > 2 else "default"
        self.device_name = device_name

        self._imu_path = self.base_dir / "imu.csv"
        self._cam_path = self.base_dir / "cam.csv"
        self._gt_path = self.base_dir / "gt.csv"
        self._opt_path = self.base_dir / "opt.csv"

        self.has_opt = self._opt_path.exists()

    def load_data(self, using_opt=False):
        self.imu_data = ImuData.from_csv(self._imu_path)
        self.gt_data = GroundTruthData.from_csv(self._gt_path)
        if self.has_opt:
            self.opt_data = GroundTruthData.from_csv(self._opt_path)
            if using_opt:
                self.gt_data = self.opt_data

        if len(self.imu_data) != len(self.gt_data):
            print("! imu and gt data length not match")


class ExtUnitData(UnitData):
    imu_data: ImuData
    gt_data: PosesData
    opt_data: PosesData
    fusion_data: PosesData
    check_data: DataCheck
    calib_data: CalibrationData

    def __init__(self, base_dir: Path | str, using_ext: bool = True):
        super().__init__(base_dir)
        self.using_ext = using_ext

        # 读取真值
        db_file = RTABData.get_db_file(self.base_dir)
        if db_file is not None:
            # 两种文件均不存在
            rtab_data = RTABData(db_file)

            rtab_data.transform_local()
            rtab_data.save_csv(self._gt_path)
            rtab_data.save_csv(self._opt_path, using_opt=True)

        self._fusion_path = self.base_dir / "fusion.csv"
        self.has_fusion = self._fusion_path.exists()

        # 读取标定数据
        if using_ext:
            self._calib_file = self.base_dir / "Calibration.json"
            self._check_file = self.base_dir / "DataCheck.json"

    def load_data(self, using_opt=False):
        super().load_data(using_opt)

        if self.has_fusion:
            fusion_data = FusionData.from_csv(self._fusion_path)
            self.fusion_data = fusion_data

        self.correct(self.gt_data, self.imu_data)

    def correct(self, gt_data: PosesData, imu_data: ImuData):
        self.calib_data = CalibrationData.from_json(self._calib_file)
        self.check_data = DataCheck.from_json(self._check_file)
        # 时间修正
        gt_data.t_us += self.check_data.t_gi_us
        # 空间变换
        gt_data.transform_local(self.calib_data.tf_sg_local.inverse())
        # gt_data.transform_global(self.calib_data.tf_sg_global)
        # gt_data.transform_global(gt_data.get_pose(0).get_yaw_pose().inverse())

        # 数据对齐
        t_new_us = get_time_series([imu_data.t_us, gt_data.t_us])
        self.imu_data = imu_data.interpolate(t_new_us)
        self.gt_data = gt_data.interpolate(t_new_us)


class DeviceDataset:
    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.device_name = self.base_dir.name
        self.units = [
            UnitData(path) for path in self.base_dir.iterdir() if path.is_dir()
        ]

    def __getitem__(self, index):
        return self.units[index]

    def __len__(self):
        return len(self.units)
