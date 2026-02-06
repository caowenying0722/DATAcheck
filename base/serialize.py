import json
from pathlib import Path

import numpy as np
import pandas as pd

from .datatype import GroundTruthData, ImuData, PosesData


class TypeHeader:
    t_us = ["timestamp [us]"]
    acce = ["a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]
    gyro = ["w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]"]
    magn = ["m_RS_S_x [µT]", "m_RS_S_y [µT]", "m_RS_S_z [µT]"]
    quat = ["q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"]
    posi = ["p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]"]
    t_sys_us = ["t_system [us]"]

    def __init__(self):
        self.header = []

    def add_custom(
        self, sign: str, unit: str, coord: str = "RS_S", dims: list = ["x", "y", "z"]
    ):
        list = [f"{sign}_{coord}_{d} [{unit}]" for d in dims]
        return self.add(list)

    def add(self, ty: list[str]):
        self.header.extend(ty)
        return self

    def finish(self):
        header = self.header
        header[0] = f"#{header[0]}"
        return header


class Serializer:
    def save(self, path: Path | str): ...


class PosesDataSerializer(Serializer):
    def __init__(self, data: PosesData):
        self.data = data

    def save(self, path: Path | str):
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        if path.exists():
            print(f"File {path} already exists.")
            return

        data = self.data
        raw = np.hstack(
            [
                data.t_us.reshape(-1, 1),
                data.ps,
                data.rots.as_quat(scalar_first=True),
            ]
        )
        header = (
            TypeHeader().add(TypeHeader.t_us).add(TypeHeader.posi).add(TypeHeader.quat)
        ).finish()
        pd.DataFrame(raw, columns=header).to_csv(path, index=False, float_format="%.6f")  # type:ignore


class ImuDataSerializer(Serializer):
    def __init__(self, data: ImuData):
        self.data = data

    def save(self, path: Path | str):
        data = self.data
        raw = np.hstack(
            [
                data.t_us.reshape(-1, 1),
                data.gyro,
                data.acce,
                data.ahrs.as_quat(scalar_first=True),
                data.t_us.reshape(-1, 1),
                data.magn,
            ]
        )
        header = (
            TypeHeader()
            .add(TypeHeader.t_us)
            .add(TypeHeader.gyro)
            .add(TypeHeader.acce)
            .add(TypeHeader.quat)
            .add(TypeHeader.t_sys_us)
            .add(TypeHeader.magn)
        ).finish()
        pd.DataFrame(raw, columns=header).to_csv(path, index=False, float_format="%.6f")  # type:ignore


class UnitSerializer(Serializer):
    def __init__(self, imu_data: ImuData, gt_data: PosesData, calibration_files=None):
        """
        Args:
            imu_data: IMU数据
            gt_data: Ground Truth数据
            calibration_files: 需要复制的校准文件列表（如 info.json, calibration.json）
        """
        self.imu_data = imu_data
        self.gt_data = gt_data
        self.calibration_files = calibration_files or []

    def save(self, path: Path | str):
        path = Path(path)
        if path.exists():
            print("Skip.")
            return

        path.mkdir(parents=True, exist_ok=True)

        imu_path = path / "imu.csv"
        gt_path = path / "gt.csv"

        ImuDataSerializer(self.imu_data).save(imu_path)
        PosesDataSerializer(self.gt_data).save(gt_path)

        # 复制校准文件
        import shutil
        for calib_file in self.calibration_files:
            calib_file = Path(calib_file)
            if calib_file.exists():
                shutil.copy2(calib_file, path / calib_file.name)
                print(f"Copied calibration: {calib_file.name}")

        print(f"Save to: {path}")


class TLIOSerializer(Serializer):
    JSON_TEMP = {
        "columns_name(width)": [
            "ts_us(1)",
            "gyr_compensated_rotated_in_World(3)",
            "acc_compensated_rotated_in_World(3)",
            "qxyzw_World_Device(4)",
            "pos_World_Device(3)",
            "vel_World(3)",
        ],
        "num_rows": 0,
        "approximate_frequency_hz": 0.0,
        "t_start_us": 0.0,
        "t_end_us": 0.0,
    }

    def __init__(self, imu_data: ImuData, gt_data: GroundTruthData):
        self.imu_data = imu_data
        self.gt_data = gt_data

    def save(self, path: Path | str):
        path = Path(path)

        if path.exists():
            print("Skip.")
            return

        path.mkdir(parents=True, exist_ok=True)
        self.npy_file = path / "imu0_resampled.npy"
        self.json_file = path / "imu0_resampled_description.json"

        world_imu = self.imu_data.transform(self.gt_data.rots)

        t_us = self.gt_data.t_us.reshape(-1, 1)
        ps = self.gt_data.ps
        _dt_s = np.diff(t_us, axis=0).reshape(-1, 1) * 1e-6
        vs = np.diff(ps, axis=0) / _dt_s
        vs = np.vstack([vs[0], vs])

        imu0_resampled = np.hstack(
            [
                t_us,
                world_imu.gyro,
                world_imu.acce,
                self.gt_data.rots.as_quat(scalar_first=True),
                ps,
                vs,
            ]
        )

        np.save(self.npy_file, imu0_resampled)

        json_info = self.JSON_TEMP.copy()
        json_info["approximate_frequency_hz"] = self.gt_data.rate
        json_info["num_rows"] = int(len(self.gt_data))
        json_info["t_start_us"] = float(self.imu_data.t_us[0])
        json_info["t_end_us"] = float(self.imu_data.t_us[-1])
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(json_info, f, indent=4)
            print(f"Save {json_info}")

        print(f"Save to: {path}")
