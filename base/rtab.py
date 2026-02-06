import sqlite3
import struct
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from base.interpolate import interpolate_vector3, slerp_rotation

# 对其IMU矩阵
DefaultBodyRotation = Rotation.from_rotvec([0, -90, 0], degrees=True)


class RTABData:
    opt_ids: list[int]
    opt_t_us: NDArray[np.int64]
    opt_rots: Rotation
    opt_ps: NDArray

    node_ids: list[int]
    node_t_us: NDArray[np.int64]
    node_rots: Rotation
    node_ps: NDArray

    t_us_f0: NDArray[np.int64]
    t_sys_us: NDArray[np.int64]

    t_len_s: float
    rate: float

    def __init__(
        self,
        file_path: str | Path,
        *,
        is_load_opt: bool = True,
    ):
        self.file_path = str(file_path)
        self.opt_ids = []
        self.node_ids = []

        if self.file_path.endswith(".csv"):
            # rtab.csv 文件
            print("Loading from CSV data...")
            self.load_csv_data()
        elif self.file_path.endswith(".db"):
            # *.db 文件
            print("Loading from DB data...")
            # 从数据库中加载
            self.conn = sqlite3.connect(self.file_path)
            self.cursor = self.conn.cursor()
            self.load_node_data()
            if is_load_opt:
                self.load_opt_data()
        else:
            raise ValueError("RTAB-Map data file must be a .csv or .db file")

        self.t_us_f0 = self.node_t_us - self.node_t_us[0]
        self.t_sys_us = self.node_t_us
        self.rate = float(1e6 / np.mean(np.diff(self.node_t_us)))
        self.t_len_s = (self.node_t_us[-1] - self.node_t_us[0]) / 1e6

    def __len__(self):
        return len(self.node_t_us)

    @classmethod
    def _decompress_data(cls, blob_data) -> bytes | None:
        """通用的解压缩和解析函数"""
        try:
            # 尝试解压缩数据
            try:
                decompressed_data = zlib.decompress(blob_data)
            except Exception:
                decompressed_data = blob_data

            return decompressed_data
        except Exception as e:
            print(f"Error decompressing data: {e}")
            return None

    @classmethod
    def _unpack_pose_data(cls, blob_data) -> tuple | None:
        """解压并解析RTAB-Map的pose数据"""
        decompressed_data = cls._decompress_data(blob_data)
        if decompressed_data and len(decompressed_data) >= 48:
            pose_values = struct.unpack("12f", decompressed_data[:48])
            pose_matrix = np.array(pose_values).reshape(3, 4)
            p = pose_matrix[:3, 3]
            rot = Rotation.from_matrix(pose_matrix[:3, :3])
            return (rot, p)
        return None

    @staticmethod
    def get_db_file(base_dir: Path):
        for file in base_dir.iterdir():
            if file.suffix == ".db":
                gt_path = file
                break
        else:
            return None
        return gt_path

    def load_csv_data(self):
        # #timestamp [us],p_RN_x [m],p_RN_y [m],p_RN_z [m],q_RN_w [],q_RN_x [],q_RN_y [],q_RN_z []
        data = pd.read_csv(self.file_path).to_numpy()
        self.node_t_us = data[:, 0]
        self.node_ps = data[:, 1:4]
        self.node_rots = Rotation.from_quat(data[:, 4:], scalar_first=True)
        self.opt_rots = self.node_rots
        self.opt_ps = self.node_ps

    def load_opt_data(self):
        print("Loading optimized data...")
        # 查询Admin表中的opt_poses数据
        admin_opt_poses = self.cursor.execute(
            "SELECT opt_poses FROM Admin WHERE opt_poses IS NOT NULL"
        ).fetchone()
        assert admin_opt_poses is not None, "Failed to fetch admin_opt_poses data"
        decompressed_data = self._decompress_data(admin_opt_poses[0])

        assert decompressed_data is not None, (
            "Failed to decompress admin_opt_poses data"
        )

        opt_rots = []
        opt_ps = []
        for pose_values in struct.iter_unpack("12f", decompressed_data):
            pose_matrix = np.array(pose_values).reshape(3, 4)
            R = pose_matrix[:3, :3]
            p = pose_matrix[:3, 3]
            opt_rots.append(R)
            opt_ps.append(p)

        self.opt_rots = Rotation.from_matrix(opt_rots)
        self.opt_ps = np.array(opt_ps)

        # 查询Admin表中的 opt_ids 数据
        admin_opt_ids = self.cursor.execute(
            "SELECT opt_ids FROM Admin WHERE opt_ids IS NOT NULL"
        ).fetchone()

        decompressed_ids = self._decompress_data(admin_opt_ids[0])
        assert decompressed_ids is not None, "Failed to decompress admin_opt_ids data"

        self.opt_ids = list(
            struct.unpack(f"{len(decompressed_ids) // 4}i", decompressed_ids)
        )
        assert (
            self.opt_ids.__len__() == self.opt_ps.__len__() == self.opt_rots.__len__()
        ), "Mismatch in lengths of opt_ids, ps, and unit_qs"

        # 通过 ids 获取时间戳
        opt_t_us = [self.node_t_us[self.node_ids.index(idx)] for idx in self.opt_ids]
        self.opt_t_us = np.array(opt_t_us)

    def load_node_data(self):
        results = self.cursor.execute("""
            SELECT id, stamp, pose
            FROM Node
            WHERE pose IS NOT NULL
            ORDER BY id
        """).fetchall()

        assert results, f"No node data found in the database. {self.file_path}"

        node_t_us = []
        node_rots = []
        node_ps = []
        for node_id, stamp, pose_blob in results:
            pose = self._unpack_pose_data(pose_blob)
            if pose is None:
                continue

            self.node_ids.append(node_id)
            node_t_us.append(int(stamp * 1e6))  # convert to us
            node_rots.append(pose[0].as_quat())
            node_ps.append(pose[1])

        self.node_t_us = np.array(node_t_us)
        self.node_rots = Rotation.from_quat(node_rots)
        self.node_ps = np.array(node_ps)

    def fix_time(self, t_21_us: int):
        self.node_t_us += t_21_us
        self.t_us_f0 += t_21_us
        self.t_sys_us += t_21_us

    def interpolate(self, t_new_us: NDArray):
        self.node_rots = slerp_rotation(self.node_rots, self.t_sys_us, t_new_us)
        self.node_ps = interpolate_vector3(self.node_ps, self.t_sys_us, t_new_us)
        self.t_sys_us = t_new_us

    def transform_local(self):
        self.node_rots = self.node_rots * DefaultBodyRotation
        self.opt_rots = self.opt_rots * DefaultBodyRotation

    def save_csv(self, path: str | Path, using_opt: bool = False):
        # #timestamp [us],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],
        header = [
            "#timestamp [us]",
            "p_RN_x [m]",
            "p_RN_y [m]",
            "p_RN_z [m]",
            "q_RN_w []",
            "q_RN_x []",
            "q_RN_y []",
            "q_RN_z []",
        ]

        t_us = self.node_t_us if not using_opt else self.opt_t_us
        ps = self.node_ps if not using_opt else self.opt_ps
        rots = self.node_rots if not using_opt else self.opt_rots
        data = np.hstack(
            [
                t_us.reshape(-1, 1),
                ps,
                rots.as_quat(scalar_first=True),
            ]
        )

        pd.DataFrame(data, columns=header).to_csv(  # type:ignore
            path, index=False, float_format="%.8f"
        )
        print(f"Saved {len(self.node_ids)} poses to {path}")
