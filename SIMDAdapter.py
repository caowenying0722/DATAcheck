"""
SIMD 数据集适配器 - 将 SIMD 格式转换为标准格式

SIMD 数据集格式说明：
- Sys_time: 系统时间戳（毫秒）
- gyrx,y,z: 陀螺仪
- accx,y,z: 加速度计
- magx,y,z: 磁力计
- rot_x,rot_y,rot_z,rot_s: 旋转四元数（局部坐标系）
- grot_x,grot_y,grot_z,g_rot_s: 全局旋转四元数（可以作为 GT）

转换策略：
1. 使用全局旋转 (grot) 作为 Ground Truth
2. 使用加速度积分估算位置（如果需要位置信息）
3. 单位转换：
   - 时间：毫秒 -> 微秒
   - 四元数：保持原样
   - 陀螺仪：rad/s（假设已是标准单位）
   - 加速度：m/s²（假设已是标准单位）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from base.datatype import ImuData, GroundTruthData, UnitData


class SIMDAdapter:
    """SIMD 数据集适配器"""

    @staticmethod
    def load(csv_path: Path):
        """
        加载单个 CSV 文件并转换为 UnitData

        Args:
            csv_path: CSV 文件路径

        Returns:
            UnitData: 标准格式的单元数据
        """
        csv_path = Path(csv_path)

        # 读取 CSV（header=0，列名格式特殊）
        df = pd.read_csv(csv_path)

        # 处理可能的重复表头行（某些文件在中间包含重复的表头）
        # 检查是否有行的第一列是 "Sys_time"
        if len(df) > 0 and df.iloc[:, 0].dtype == 'object':
            # 找到所有第一列不是 "Sys_time" 的行
            mask = df.iloc[:, 0] != 'Sys_time'
            df = df[mask].reset_index(drop=True)

        # 确保 Sys_time 列是数值类型
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')

        # SIMD 数据集的列名格式特殊，例如：gyrx,y,z 而不是 gyrx, gyry, gyrz
        # 由于列名重复，pandas 会自动添加后缀：y.1, y.2, z.1, z.2 等
        # 列索引（0-based）：
        # 0: Sys_time
        # 1-3: laccx, y, z
        # 4: lacc_accu
        # 5-7: grax, y.1, z.1
        # 8: gra_accu
        # 9-11: gyrx, y.2, z.2  ← 陀螺仪
        # 12: gyr_accu
        # 13-15: accx, y.3, z.3  ← 加速度
        # 16: acc_accu
        # 17-19: magx, y.4, z.4  ← 磁力计
        # 20: mag_accu
        # 21: ori
        # 22-25: rot_x, rot_y, rot_z, rot_s  ← 局部旋转
        # 26-27: rot_head_acc, rot_accu
        # 28-31: grot_x, grot_y, grot_z, g_rot_s  ← 全局旋转
        # 32: g_rot_accu
        # 33-38: lon, lat, speed, bearing, gps_time, step

        # ==================== 1. 时间转换 ====================
        # Sys_time 是毫秒，转换为微秒
        t_us = (df.iloc[:, 0].values * 1000).astype(np.int64)

        # ==================== 2. 陀螺仪数据 ====================
        # gyrx, y.2, z.2 位于第9-11列（索引9,10,11）
        gyro = df.iloc[:, 9:12].values.astype(np.float64)

        # ==================== 3. 加速度数据 ====================
        # accx, y.3, z.3 位于第13-15列（索引13,14,15）
        acce = df.iloc[:, 13:16].values.astype(np.float64)

        # ==================== 4. 姿态数据 (AHRS) ====================
        # rot_x, rot_y, rot_z, rot_s 位于第22-25列（索引22,23,24,25）
        # 注意：SIMD 的格式是 (x, y, z, s/w)
        if df.shape[1] >= 26:
            # rot_s 在索引25，rot_x,rot_y,rot_z 在索引22,23,24
            quat_ahrs = df.iloc[:, [25, 22, 23, 24]].values  # s, x, y, z (scalar first)
            # 确保 scalar first
            ahrs = Rotation.from_quat(quat_ahrs, scalar_first=True)
        else:
            # 如果没有旋转数据，使用单位四元数
            ahrs = Rotation.identity(len(t_us))

        # ==================== 5. 磁力计数据 ====================
        # magx, y.4, z.4 位于第17-19列（索引17,18,19）
        if df.shape[1] >= 20:
            magn = df.iloc[:, 17:20].values.astype(np.float64)
        else:
            magn = np.zeros((len(t_us), 3), dtype=np.float64)

        # ==================== 6. 创建 IMU 数据 ====================
        imu_data = ImuData(
            t_us=t_us,
            gyro=gyro,
            acce=acce,
            ahrs=ahrs,
            magn=magn,
            frame="local"
        )

        # ==================== 7. 创建 Ground Truth 数据 ====================
        # 使用全局旋转 (grot) 作为 Ground Truth
        # grot_x, grot_y, grot_z, g_rot_s 位于第28-31列（索引28,29,30,31）
        if df.shape[1] >= 32:
            # g_rot_s 在索引31，grot_x,grot_y,grot_z 在索引28,29,30
            quat_gt = df.iloc[:, [31, 28, 29, 30]].values  # s, x, y, z (scalar first)
            rots = Rotation.from_quat(quat_gt, scalar_first=True)
        else:
            # 如果没有全局旋转，使用局部旋转
            rots = ahrs

        # 位置信息：如果有 GPS 数据，可以转换
        # lon, lat 位于第33-34列（索引33,34）
        if df.shape[1] >= 35:
            # 尝试将经纬度转换为局部坐标
            # 简单方法：以第一个点为原点，计算相对的东北天(ENU)坐标
            lon = df.iloc[:, 33].values
            lat = df.iloc[:, 34].values

            # 过滤掉 null 值
            valid_mask = (lon != 'null') & (lat != 'null')

            if np.any(valid_mask):
                # 有有效的 GPS 数据
                lon_vals = np.array([float(x) if x != 'null' else np.nan for x in lon])
                lat_vals = np.array([float(x) if x != 'null' else np.nan for x in lat])

                # 找到第一个有效点作为参考点
                first_valid_idx = np.where(~np.isnan(lon_vals))[0][0]
                ref_lon = lon_vals[first_valid_idx]
                ref_lat = lat_vals[first_valid_idx]

                # 转换为相对 ENU 坐标（简化版本）
                # 1度经度约等于 111km * cos(lat)
                # 1度纬度约等于 111km
                earth_radius = 6371000.0  # 地球半径（米）
                lat_rad = np.radians(ref_lat)

                ps = np.zeros((len(t_us), 3), dtype=np.float64)
                for i in range(len(t_us)):
                    if not np.isnan(lon_vals[i]) and not np.isnan(lat_vals[i]):
                        # 东向距离（米）
                        ps[i, 0] = (lon_vals[i] - ref_lon) * earth_radius * np.cos(lat_rad) * np.pi / 180
                        # 北向距离（米）
                        ps[i, 1] = (lat_vals[i] - ref_lat) * earth_radius * np.pi / 180
                        # 高度（设为0，因为没有数据）
                        ps[i, 2] = 0.0
            else:
                # 没有 GPS 数据，使用零位置
                ps = np.zeros((len(t_us), 3), dtype=np.float64)
        else:
            ps = np.zeros((len(t_us), 3), dtype=np.float64)

        gt_data = GroundTruthData(
            t_us=t_us,
            rots=rots,
            ps=ps
        )

        # ==================== 8. 封装为 UnitData ====================
        unit = UnitData.__new__(UnitData)
        unit.name = csv_path.stem  # 使用文件名作为序列名
        unit.base_dir = csv_path.parent
        unit.imu_data = imu_data
        unit.gt_data = gt_data

        return unit


def convert_simd_to_standard(simd_root: Path, output_root: Path, split: str = 'all'):
    """
    批量转换 SIMD 数据集为标准格式

    Args:
        simd_root: SIMD 数据集根目录
        output_root: 输出目录
        split: 数据集分割 ('train', 'val', 'test', 'all')
    """
    from base.serialize import ImuDataSerializer, PosesDataSerializer

    simd_root = Path(simd_root)
    output_root = Path(output_root)

    # 确定要处理的文件列表
    if split == 'all':
        split_file = simd_root / 'all'
        csv_files = list(split_file.glob('*.csv'))
    else:
        split_file = simd_root / f'{split}.txt'
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            file_names = [line.strip() for line in f if line.strip()]

        csv_files = [(simd_root / 'all' / fname) for fname in file_names]

    print(f"📁 Found {len(csv_files)} files in {split} split")

    # 批量转换
    success_count = 0
    for i, csv_file in enumerate(csv_files, 1):
        try:
            print(f"[{i}/{len(csv_files)}] Converting: {csv_file.name}")

            # 加载并转换
            unit = SIMDAdapter.load(csv_file)

            # 创建输出目录
            output_dir = output_root / unit.name
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存为标准格式
            ImuDataSerializer(unit.imu_data).save(output_dir / 'imu.csv')
            PosesDataSerializer(unit.gt_data).save(output_dir / 'gt.csv')

            success_count += 1

        except Exception as e:
            print(f"  ❌ Failed: {e}")

    print(f"\n✅ Conversion complete: {success_count}/{len(csv_files)} files")
    return success_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert SIMD dataset to standard format")
    parser.add_argument("--simd-root", type=str, default="/home/vln/imuproject/MdlVerifyV1/imudata/SIMD",
                        help="SIMD dataset root directory")
    parser.add_argument("--output-root", type=str, required=True,
                        help="Output directory for standard format")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"],
                        help="Dataset split to convert")

    args = parser.parse_args()

    print("=" * 80)
    print("🔄 SIMD Dataset Converter")
    print("=" * 80)
    print(f"Input:  {args.simd_root}")
    print(f"Output: {args.output_root}")
    print(f"Split:  {args.split}")
    print("=" * 80)

    convert_simd_to_standard(args.simd_root, args.output_root, args.split)
