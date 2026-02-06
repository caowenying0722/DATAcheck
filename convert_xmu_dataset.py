"""
XMU 数据集预处理脚本

本模块用于将 XMU 数据集预处理并转换为项目标准化的数据格式。

=== 数据集信息 ===
数据集名称：XMU Dataset
厦门大学 IMU 数据集

=== 数据集结构 ===
每个序列包含以下文件：
- imu.csv: IMU数据 (timestamp, gyro, accel, quat, t_system)
- rtab.csv: RTAB-Map地面真值轨迹数据
- Calibration.json: 传感器与参考坐标系之间的标定信息
- DataCheck.json: 时间同步偏移信息

=== 预处理步骤 ===
1. 读取原始 IMU 数据和 RTAB-Map 地面真值数据
2. 从 Calibration.json 加载硬件标定参数（旋转和平移）
3. 从 DataCheck.json 加载时间偏移参数
4. 应用时间同步修正
5. 应用硬件校准（旋转对齐）
6. 使用 GT 姿态转换到世界坐标系
7. 重采样和插值完成时间同步对齐
8. 保存为标准数据格式

=== 使用示例 ===
    # 转换单个数据集文件夹
    uv run python imudata/XMU/convert_xmu_dataset.py -d /path/to/XMU/1

    # 转换所有 XMU 数据集
    uv run python imudata/XMU/convert_xmu_dataset.py -d /path/to/XMU --all
"""

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# 导入 base 库相关工具
import sys
# 添加 base 目录到 Python 路径（base 在当前目录下）
_base_path = Path(__file__).parent / "base"
if _base_path.exists():
    sys.path.insert(0, str(_base_path))

from base.interpolate import get_time_series, interpolate_vector3, slerp_rotation
from base.datatype import GroundTruthData, ImuData, PosesData, Pose
from base.serialize import UnitSerializer
from base.args_parser import DatasetArgsParser


class XMUCalibration:
    """XMU 数据集标定信息"""

    def __init__(self, calib_json_path: Path):
        with open(calib_json_path, "r") as f:
            data = json.load(f)
            if not isinstance(data, list) or len(data) != 1:
                raise ValueError(f"Invalid Calibration.json format: {calib_json_path}")
            data = data[0]

        # rot_sensor_gt: 传感器到地面真值的旋转
        rot_sensor_gt = np.array(data["rot_sensor_gt"])
        self.rot_sensor_gt = Rotation.from_matrix(rot_sensor_gt)

        # trans_sensor_gt: 传感器到地面真值的平移
        self.trans_sensor_gt = np.array(data["trans_sensor_gt"]).flatten()

        # rot_ref_sensor_gt: 参考系到传感器的旋转
        rot_ref_sensor_gt = np.array(data["rot_ref_sensor_gt"])
        self.rot_ref_sensor_gt = Rotation.from_matrix(rot_ref_sensor_gt)

        # trans_ref_sensor_gt: 参考系到传感器的平移
        self.trans_ref_sensor_gt = np.array(data["trans_ref_sensor_gt"]).flatten()

    def get_sensor_to_gt_transform(self):
        """获取传感器到地面真值的变换"""
        return Pose(self.rot_sensor_gt, self.trans_sensor_gt)

    def get_ref_to_sensor_transform(self):
        """获取参考系到传感器的变换"""
        return Pose(self.rot_ref_sensor_gt, self.trans_ref_sensor_gt)


class XMUDataCheck:
    """XMU 数据集时间同步信息"""

    def __init__(self, check_json_path: Path):
        with open(check_json_path, "r") as f:
            data = json.load(f)

        self.time_diff_21_us = data["check_time_diff"]["time_diff_21_us"]


class XMUSequence:
    """XMU 单个序列数据处理器"""

    def __init__(self, sequence_dir: Path):
        self.sequence_dir = Path(sequence_dir)
        self.sequence_name = self.sequence_dir.name

        # 数据文件路径
        self.imu_csv = self.sequence_dir / "imu.csv"
        self.rtab_csv = self.sequence_dir / "rtab.csv"
        self.calib_json = self.sequence_dir / "Calibration.json"
        self.check_json = self.sequence_dir / "DataCheck.json"

        # 检查必要文件是否存在
        if not all([
            self.imu_csv.exists(),
            self.rtab_csv.exists(),
            self.calib_json.exists(),
            self.check_json.exists()
        ]):
            missing = []
            if not self.imu_csv.exists():
                missing.append("imu.csv")
            if not self.rtab_csv.exists():
                missing.append("rtab.csv")
            if not self.calib_json.exists():
                missing.append("Calibration.json")
            if not self.check_json.exists():
                missing.append("DataCheck.json")
            raise ValueError(f"Missing required files: {missing}")

        # 加载标定和时间同步信息
        self.calib = XMUCalibration(self.calib_json)
        self.check = XMUDataCheck(self.check_json)

    def load_imu_data(self) -> ImuData:
        """加载 IMU 数据

        XMU imu.csv 格式:
        - Column 0: timestamp [us] (传感器时间)
        - Column 1-3: gyro (rad/s)
        - Column 4-6: accel (m/s^2)
        - Column 7-10: orientation quaternion (w,x,y,z)
        - Column 11: t_system [us] (系统时间，与 GT 时间对齐)

        注意：我们需要使用 t_system 作为时间戳，以与 GT 数据对齐
        """
        raw = pd.read_csv(self.imu_csv).to_numpy()
        # 使用系统时间 (t_system, column 11) 作为时间戳，以便与 GT 对齐
        t_us = raw[:, 11].astype(np.int64)
        gyro = raw[:, 1:4]
        acce = raw[:, 4:7]
        ahrs = Rotation.from_quat(raw[:, 7:11], scalar_first=True)
        # 如果有磁场数据（第12列之后），使用它，否则填充零
        if raw.shape[1] >= 15:
            magn = raw[:, 12:15]
        else:
            magn = np.zeros_like(gyro)

        return ImuData(t_us, gyro, acce, ahrs, magn, frame="local")

    def load_gt_data(self) -> GroundTruthData:
        """加载地面真值数据 (rtab.csv)"""
        raw = pd.read_csv(self.rtab_csv).to_numpy()
        # rtab.csv: timestamp, pos(x,y,z), quat(w,x,y,z)
        t_us = raw[:, 0].astype(np.int64)
        pos = raw[:, 1:4]
        qua = raw[:, 4:8]
        rots = Rotation.from_quat(qua, scalar_first=True)

        return GroundTruthData(t_us, rots, pos)

    def preprocess(self) -> tuple[ImuData, GroundTruthData]:
        """
        预处理 XMU 数据序列

        步骤：
        1. 加载原始数据
        2. 应用时间同步修正
        3. 应用硬件校准（旋转对齐）
        4. 使用 GT 姿态转到世界系
        5. 重采样和插值对齐
        """
        # 1. 加载原始数据
        print(f"  Loading IMU data from {self.imu_csv.name}...")
        imu_data = self.load_imu_data()
        imu_rate = float(1e6 / np.mean(np.diff(imu_data.t_us)))
        print(f"    IMU: {len(imu_data)} samples, rate: {imu_rate:.2f} Hz")

        print(f"  Loading GT data from {self.rtab_csv.name}...")
        gt_data = self.load_gt_data()
        gt_rate = float(1e6 / np.mean(np.diff(gt_data.t_us)))
        print(f"    GT: {len(gt_data)} samples, rate: {gt_rate:.2f} Hz")

        # 2. 应用时间同步修正
        print(f"  Applying time sync offset: {self.check.time_diff_21_us} us")
        gt_data.t_us += self.check.time_diff_21_us

        # 3. 应用硬件校准（旋转对齐）
        # 利用设备的先验角度差，完成硬件校准
        print(f"  Applying hardware calibration (rotation alignment)...")
        # 使用传感器到地面真值的变换的逆变换
        sensor_to_gt_tf = self.calib.get_sensor_to_gt_transform()
        gt_data.transform_local(sensor_to_gt_tf.inverse())

        # 4. 使用 GT 姿态转到世界系
        # 首先找到 IMU 和 GT 时间的重叠范围
        print(f"  Finding overlapping time range...")
        t_start_us = max(imu_data.t_us[0], gt_data.t_us[0])
        t_end_us = min(imu_data.t_us[-1], gt_data.t_us[-1])
        print(f"    IMU time range: [{imu_data.t_us[0]}, {imu_data.t_us[-1]}]")
        print(f"    GT time range: [{gt_data.t_us[0]}, {gt_data.t_us[-1]}]")
        print(f"    Overlap: [{t_start_us}, {t_end_us}]")

        # 裁剪 IMU 数据到重叠范围
        imu_mask = (imu_data.t_us >= t_start_us) & (imu_data.t_us <= t_end_us)
        imu_data_cropped = ImuData(
            imu_data.t_us[imu_mask],
            imu_data.gyro[imu_mask],
            imu_data.acce[imu_mask],
            imu_data.ahrs[imu_mask],
            imu_data.magn[imu_mask],
            frame="local"
        )
        print(f"    IMU cropped: {len(imu_data)} -> {len(imu_data_cropped)} samples")

        # 将 GT 数据插值到裁剪后的 IMU 时间戳
        print(f"  Interpolating GT data to IMU timestamps...")
        gt_interpolated = gt_data.interpolate(imu_data_cropped.t_us, bounds_error=False)

        # 使用插值后的 GT 旋转将 IMU 数据转换到世界坐标系
        print(f"  Transforming IMU to world frame using GT orientation...")
        imu_data_world = imu_data_cropped.transform(gt_interpolated.rots)

        # 5. 重采样和插值对齐
        print(f"  Resampling and interpolating to aligned timestamps...")
        # 计算公共时间序列（使用统一的采样率）
        t_new_us = get_time_series([imu_data_world.t_us, gt_data.t_us])
        aligned_rate = 1e6 / np.mean(np.diff(t_new_us))
        print(f"    Aligned to {len(t_new_us)} samples, rate: {aligned_rate:.2f} Hz")

        # 插值对齐
        imu_aligned = imu_data_world.interpolate(t_new_us, bounds_error=False)
        gt_aligned = gt_data.interpolate(t_new_us, bounds_error=False)

        return imu_aligned, gt_aligned

    def save(self, output_dir: Path):
        """预处理并保存数据"""
        print(f"\nProcessing sequence: {self.sequence_name}")

        # 预处理
        imu_data, gt_data = self.preprocess()

        # 保存到标准格式
        output_sequence_dir = output_dir / self.sequence_name
        UnitSerializer(imu_data, gt_data).save(output_sequence_dir)

        print(f"  Saved to: {output_sequence_dir}")
        return output_sequence_dir


class XMUDatasetConverter:
    """XMU 数据集转换器"""

    def __init__(self, dataset_base_dir: Path | str, output_base_dir: Path | str | None = None):
        self.dataset_base_dir = Path(dataset_base_dir)

        if output_base_dir is None:
            # 默认输出到数据集同级目录
            self.output_base_dir = self.dataset_base_dir.parent / f"{self.dataset_base_dir.name}_processed"
        else:
            self.output_base_dir = Path(output_base_dir)

    def convert_sequence(self, sequence_dir: Path):
        """转换单个序列"""
        try:
            sequence = XMUSequence(sequence_dir)
            return sequence.save(self.output_base_dir)
        except Exception as e:
            print(f"  Error processing {sequence_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def convert_device_folder(self, device_folder: Path):
        """转换设备文件夹（如 2025_SM-G9900_in）下的所有序列"""
        print(f"\nConverting device folder: {device_folder.name}")

        # 查找所有序列目录（包含 imu.csv 的目录）
        sequence_dirs = [d for d in device_folder.iterdir() if d.is_dir() and (d / "imu.csv").exists()]

        print(f"  Found {len(sequence_dirs)} sequences")

        converted = []
        for seq_dir in sorted(sequence_dirs):
            result = self.convert_sequence(seq_dir)
            if result:
                converted.append(result)

        print(f"  Converted {len(converted)}/{len(sequence_dirs)} sequences")
        return converted

    def convert_dataset_folder(self, dataset_folder: Path):
        """转换数据集文件夹（如 1, 2, 3）下的所有设备"""
        print(f"\nConverting dataset folder: {dataset_folder.name}")

        # 查找所有设备文件夹（包含 imu.csv 序列的子目录）
        device_folders = []
        for d in dataset_folder.iterdir():
            if d.is_dir():
                # 检查是否包含序列目录（imu.csv）
                # 注意：seq 已经是完整路径，不需要再拼接 d
                has_sequences = any((seq / "imu.csv").exists() for seq in d.iterdir() if seq.is_dir())
                if has_sequences:
                    device_folders.append(d)

        if not device_folders:
            # 如果没有找到设备文件夹，尝试直接查找序列目录
            device_folders = [
                d for d in dataset_folder.iterdir()
                if d.is_dir() and (d / "imu.csv").exists()
            ]

        print(f"  Found {len(device_folders)} device folders: {[d.name for d in device_folders]}")

        all_converted = []
        for device_folder in sorted(device_folders):
            results = self.convert_device_folder(device_folder)
            all_converted.extend(results)

        return all_converted

    def convert_all(self):
        """转换数据集根目录下的所有文件夹"""
        print(f"\nConverting all XMU datasets in: {self.dataset_base_dir}")

        # 查找所有数据集文件夹（1, 2, 3 等）
        dataset_folders = [d for d in self.dataset_base_dir.iterdir() if d.is_dir() and d.name.isdigit()]

        print(f"Found {len(dataset_folders)} dataset folders")

        all_converted = []
        for dataset_folder in sorted(dataset_folders, key=lambda x: int(x.name)):
            results = self.convert_dataset_folder(dataset_folder)
            all_converted.extend(results)

        return all_converted


def main():
    parser = argparse.ArgumentParser(
        description="预处理 XMU 数据集并转换为项目标准格式",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="XMU 数据集路径（可以是序列目录、设备文件夹、数据集文件夹或根目录）"
    )
    parser.add_argument(
        "-g",
        "--group",
        type=str,
        default=None,
        help="数据组路径（设备文件夹）"
    )
    parser.add_argument(
        "-u",
        "--unit",
        type=str,
        default=None,
        help="数据单元路径（单个序列）"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出目录路径"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="转换数据集根目录下的所有文件夹"
    )

    args = parser.parse_args()

    # 确定输入路径
    input_path = None
    if args.unit:
        input_path = Path(args.unit)
    elif args.group:
        input_path = Path(args.group)
    else:
        input_path = Path(args.dataset)

    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        return

    # 确定输出路径
    output_path = args.output
    if output_path is None:
        # 默认输出到 imudata/changed/XMU/ 下
        dataset_name = input_path.name
        output_path = f"imudata/changed/XMU/{dataset_name}"

    # 创建转换器
    converter = XMUDatasetConverter(input_path, output_path)

    # 根据输入路径类型选择转换方式
    if args.all or (input_path.is_dir() and any((input_path / d).is_dir() and d.name.isdigit() for d in input_path.iterdir())):
        # 转换所有数据集（根目录包含 1, 2, 3 等数字文件夹）
        results = converter.convert_all()
    elif (input_path / "imu.csv").exists():
        # 单个序列目录
        results = [converter.convert_sequence(input_path)]
    elif any((input_path / d / "imu.csv").exists() for d in input_path.iterdir() if d.is_dir()):
        # 设备文件夹（group）
        results = converter.convert_device_folder(input_path)
    else:
        # 数据集文件夹（dataset）
        results = converter.convert_dataset_folder(input_path)

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Output directory: {converter.output_base_dir}")
    print(f"Total sequences converted: {len([r for r in results if r is not None])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
