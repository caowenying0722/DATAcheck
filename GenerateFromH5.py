"""
MEIO-Net 数据集转换工具

本模块用于将 MEIO-Net 数据集从 HDF5 格式转换为项目标准化的数据格式。

=== 数据集信息 ===
数据集名称：MEIO-Net Dataset
数据集地址：https://zenodo.org/records/18149105
相关论文：https://ieeexplore.ieee.org/document/11251007

引用信息：
X. Sun et al., "MEIO-Net: A Motion-Aware Early-Exit Inertial Odometry Network for Efficient
Pedestrian Dead Reckoning (PDR)," in IEEE Internet of Things Journal, doi: 10.1109/JIOT.2025.3633690.

关键词：行人航位推算（PDR）、惯性导航、物联网、深度学习、计算效率

=== 数据集结构 ===
HDF5 文件包含以下分组：
- train: 训练数据集
- test: 测试数据集
- valid: 验证数据集
- para: 参数配置（如果有）

每个分组包含多个序列，每个序列包含以下字段：
- acc: 加速度计数据 (m/s²)
- acc_bias: 加速度计偏差
- gyr: 陀螺仪数据 (rad/s)
- gyr_bias: 陀螺仪偏差
- mag: 磁力计数据
- motion_qua: 运动质量指标
- pos: 位置信息 (m)
- qua: 姿态四元数 (scalar-first format: w, x, y, z)
- ts: 时间戳 (s)

=== 转换功能 ===
本模块提供以下功能：
1. 从 HDF5 文件中读取 IMU 数据和地面真值数据
2. 自动去除传感器偏差（可选）
3. 将时间戳转换为微秒格式（uint64）
4. 将四元数转换为 scipy Rotation 对象
5. 保存为两种格式：
   - Base 格式：项目基础数据格式
   - TLIO 格式：TLIO 算法专用格式

=== 使用示例 ===
    # 转换单个文件
    loader = H5Loader("/path/to/dataset.h5")
    loader.convert_all()

    # 生成的文件结构：
    # /path/to/dataset/
    #   ├── train/
    #   │   ├── sequence_1/
    #   │   └── sequence_2/
    #   ├── test/
    #   └── valid/
    #
    # /path/to/TLIO_dataset/
    #   ├── train/
    #   ├── test/
    #   └── valid/

=== 注意事项 ===
1. 时间戳从秒转换为微秒（乘以 1e6）
2. 四元数使用 scalar-first 格式 (w, x, y, z)
3. 默认启用偏差移除（remove_bias=True）
4. 生成的目录结构与原始 HDF5 文件的分组结构一致
"""

from pathlib import Path

import h5py
import numpy as np
from h5py import Group
from scipy.spatial.transform import Rotation

from base.datatype import GroundTruthData, ImuData
from base.serialize import TLIOSerializer, UnitSerializer

sample = "/Users/qi/Resources/SimpleDemo.h5"
main = "/Users/qi/Resources/YT_server_200Hz.h5"

# 文件结构
# 'para', 'test', 'train', 'valid'
# "mid360_2025-10-27_221831_00"
# 'acc', 'acc_bias', 'gyr', 'gyr_bias', 'mag', 'motion_qua', 'pos', 'qua', 'ts'


class H5Loader:
    def __init__(self, filepath: str) -> None:
        self.filepath = Path(filepath)
        self.h5_data = h5py.File(filepath, "r")

        self.savepath = self.filepath.parent / self.filepath.name.split(".")[0]

    def _parse_unit(self, unit: Group, remove_bias=False):
        """
        <KeysViewHDF5 ['acc', 'acc_bias', 'gyr', 'gyr_bias', 'mag', 'motion_qua', 'pos', 'qua', 'ts']>
        """
        acc = np.array(unit["acc"])
        _acc_bias = np.array(unit["acc_bias"])
        gyr = np.array(unit["gyr"])
        _gyr_bias = np.array(unit["gyr_bias"])
        mag = np.array(unit["mag"])
        _motion_qua = np.array(unit["motion_qua"])
        pos = np.array(unit["pos"])
        qua = np.array(unit["qua"])
        ts = np.array(unit["ts"])

        if remove_bias:
            acc = acc - _acc_bias
            gyr = gyr - _gyr_bias

        t_us = (ts * 1e6).astype(np.uint64)
        rots = Rotation.from_quat(qua, scalar_first=True)

        imu_data = ImuData(t_us, gyr, acc, rots, mag, "local")
        gt_data = GroundTruthData(t_us, rots, pos)
        return imu_data, gt_data

    def convert(self, group: Group):
        for key in group:
            assert isinstance(key, str)
            print(key, end=": ")
            # 访问数据
            unit = group[key]
            assert isinstance(unit, Group)
            # 解析数据
            data = self._parse_unit(unit, remove_bias=True)
            # 保存数据
            base_unit_dir = self.base_group_dir / key
            tlio_unit_dir = self.tlio_group_dir / key
            UnitSerializer(*data).save(base_unit_dir)
            TLIOSerializer(*data).save(tlio_unit_dir)

    def convert_all(self):
        self.base_dir = self.savepath
        self.tlio_dir = self.savepath.parent / f"TLIO_{self.savepath.name}"
        keys = ["train", "test", "valid"]
        for key in keys:
            self.base_group_dir = self.base_dir / key
            self.tlio_group_dir = self.tlio_dir / key

            group = self.h5_data[key]
            assert isinstance(group, Group)
            self.convert(group)


if __name__ == "__main__":
    # 探索文件结构
    h5_loader = H5Loader(main)
    h5_loader.convert_all()
