import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from base.datatype import ImuData, GroundTruthData, UnitData

class StandardAdapter:
    """
    读取标准格式的 imu.csv 和 gt.csv
    """
    @staticmethod
    def load(folder_path: Path):
        folder_path = Path(folder_path)
        imu_path = folder_path / "imu.csv"

        # 支持多种 GT 文件名：gt.csv 或 rtab.csv
        gt_path = folder_path / "gt.csv"
        rtab_path = folder_path / "rtab.csv"

        # 确定使用哪个 GT 文件
        if gt_path.exists():
            gt_file = gt_path
        elif rtab_path.exists():
            gt_file = rtab_path
        else:
            raise FileNotFoundError(f"Missing gt.csv or rtab.csv in {folder_path}")

        if not imu_path.exists():
            raise FileNotFoundError(f"Missing imu.csv in {folder_path}")

        # 1. 读取 IMU
        try:
            # 假设标准格式带有 Header
            imu_data = ImuData.from_csv(imu_path)
        except Exception as e:
            raise ValueError(f"Failed to load IMU: {e}")

        # 2. 读取 GT
        try:
            gt_data = GroundTruthData.from_csv(gt_file)
        except Exception as e:
            raise ValueError(f"Failed to load GT ({gt_file.name}): {e}")

        # 3. 封装
        unit = UnitData.__new__(UnitData)
        unit.name = folder_path.name
        unit.base_dir = folder_path
        unit.imu_data = imu_data
        unit.gt_data = gt_data
        
        return unit