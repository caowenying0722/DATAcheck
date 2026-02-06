#!/usr/bin/env python3
"""
数据集验证数据生成脚本

功能说明:
    该脚本用于从原始数据集中生成经过预处理的验证数据集。主要功能包括:

    1. 数据加载与同步:
       - 从原始CSV文件加载IMU、相机、Ground Truth(GT)和优化轨迹(OPT)数据
       - 计算各数据源的共同时间窗口，并进行插值对齐
       - 通过时间匹配算法校准GT数据与IMU数据的时间戳

    2. 数据标准化:
       - 将所有数据插值到统一的时间序列
       - 重置时间起点，使数据从t=0开始
       - 输出时间校准偏移量

    3. 数据保存:
       - 保存处理后的IMU数据 (imu.csv)
       - 保存Ground Truth轨迹 (gt.csv)
       - 保存优化轨迹 (opt.csv)
       - 保存相机数据 (cam.csv)
       - 生成GT与OPT的2D轨迹对比图 (gt.png)

    4. 支持两种运行模式:
       - 单元模式: 处理指定的单个数据单元
       - 数据集模式: 批量处理整个数据集目录下的所有数据单元

使用示例:
    # 处理单个单元
    python GenerateValid.py --unit <unit_path> --output <output_dir>

    # 批量处理数据集
    python GenerateValid.py --dataset <dataset_path> --output <output_dir>

    # 指定模型 (参数已定义但当前未使用)
    python GenerateValid.py --models model1 model2 --models_path <models_dir>

输入数据格式:
    - 需要包含以下CSV文件: imu.csv, cam.csv, gt.csv, opt.csv
    - 数据应包含时间戳和相应的位姿/测量信息

输出结果:
    output/
    ├── <device_name>/
    │   ├── <unit_name>/
    │   │   ├── imu.csv      # 插值对齐后的IMU数据
    │   │   ├── gt.csv       # 时间校准后的Ground Truth
    │   │   ├── opt.csv      # 重置起点后的优化轨迹
    │   │   ├── cam.csv      # 相机数据
    │   │   └── gt.png       # GT与OPT的2D轨迹对比图
"""

from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from base.args_parser import DatasetArgsParser
from base.calibration import time
from base.datatype import (
    CameraData,
    GroundTruthData,
    ImuData,
    UnitData,
)
from base.draw.Poses import draw_trajectory_2d_compare
from base.interpolate import get_time_series
from base.serialize import ImuDataSerializer, PosesDataSerializer


def main():
    dap = DatasetArgsParser()
    dap.parser.add_argument("-m", "--models", nargs="+", help="模型")
    dap.parser.add_argument("--models_path", type=str, help="模型文件夹")
    dap.parse()

    # regen_fusion = dap.regen
    assert dap.output is not None
    output = Path(dap.output)

    def action(ud: UnitData):
        imu_data = ImuData.from_csv(ud._imu_path)
        cam_data = CameraData.from_csv(ud._cam_path)
        gt_data = GroundTruthData.from_csv(ud._gt_path)
        opt_data = GroundTruthData.from_csv(ud._opt_path)

        # 获取共同的时间窗口
        ts_us = []
        ts_us.append(gt_data.t_us)
        ts_us.append(imu_data.t_us)
        ts_us.append(opt_data.t_us)

        t_new_us = get_time_series(ts_us)
        ud.gt_data = gt_data.interpolate(t_new_us)
        ud.imu_data = imu_data.interpolate(t_new_us)
        ud.opt_data = opt_data.interpolate(t_new_us)

        # 对齐
        time_gc = time.match21(ud.imu_data.to_poses(), ud.gt_data)
        ud.gt_data.t_us += time_gc
        ud.gt_data.reset_start()
        ud.opt_data.reset_start()
        print(f"时间差：{time_gc}")

        # 保存
        base_dir = output / ud.device_name / ud.name
        base_dir.mkdir(parents=True, exist_ok=True)
        ImuDataSerializer(ud.imu_data).save(base_dir / "imu.csv")
        PosesDataSerializer(ud.gt_data).save(base_dir / "gt.csv")
        PosesDataSerializer(ud.opt_data).save(base_dir / "opt.csv")
        # cam
        PosesDataSerializer(cam_data).save(base_dir / "cam.csv")

        ax = draw_trajectory_2d_compare(
            [ud.gt_data, ud.opt_data],
            ["GT", "OPT"],
            show=False,
        )
        fig = ax.figure
        assert isinstance(fig, Figure)
        png_path = base_dir / "gt.png"
        if not png_path.exists():
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("保存成功: ", png_path)

    if dap.unit:
        ud = UnitData(dap.unit)
        action(ud)
    elif dap.dataset:
        dataset_path = Path(dap.dataset)

        dirs = [dir for dir in dataset_path.iterdir() if dir.is_dir()]
        for dir in dirs:
            units = [unit for unit in dir.iterdir() if unit.is_dir()]
            for unit in units:
                print(f"处理单元：{unit}")
                ud = UnitData(unit)
                action(ud)


if __name__ == "__main__":
    main()
