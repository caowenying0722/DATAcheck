#!/usr/bin/env python3
"""
只有真值和IMU数据时，使用此脚本进行验证。
"""

import base.rerun_ext as bre
from base.args_parser import DatasetArgsParser
from base.datatype import (
    DeviceDataset,
    GroundTruthData,
    ImuData,
    PosesData,
    UnitData,
)
from base.evaluate import Evaluation
from base.interpolate import get_time_series
from base.model import DataRunner, InertialNetworkData, ModelLoader


def main():
    dap = DatasetArgsParser()
    dap.parser.add_argument("-m", "--models", nargs="+", help="模型")
    dap.parser.add_argument("--models_path", type=str, help="模型文件夹")
    dap.parse()

    # regen_fusion = dap.regen
    model_names = dap.args.models
    if model_names is None or len(model_names) == 0:
        model_names = ["model_resnet_0111_96"]

    models_path = "models"
    if dap.args.models_path is not None:
        models_path = dap.args.models_path

    loader = ModelLoader(models_path)
    Data = InertialNetworkData.set_step(20)
    nets = loader.get_by_names(model_names)

    def action(ud: UnitData):
        imu_data = ImuData.from_csv(ud._imu_path)
        gt_data = GroundTruthData.from_csv(ud._gt_path)

        # 获取共同的时间窗口
        ts_us = []
        ts_us.append(gt_data.t_us)
        ts_us.append(imu_data.t_us)

        t_new_us = get_time_series(ts_us)
        ud.gt_data = gt_data.interpolate(t_new_us)
        ud.imu_data = imu_data.interpolate(t_new_us)

        # 获取共同的时间窗口
        ts_us = []
        ts_us.append(gt_data.t_us)
        ts_us.append(imu_data.t_us)

        t_new_us = get_time_series(ts_us)
        ud.gt_data = gt_data.interpolate(t_new_us)
        ud.imu_data = imu_data.interpolate(t_new_us)

        # 可视化
        bre.RerunView().add_spatial_view().send(ud.name)
        bre.send_pose_data(ud.gt_data, "Groundtruth", color=[192, 72, 72])

        # 模型推理
        runner = DataRunner(ud, Data, using_gt=True, has_init_rerun=True)
        net_results = runner.predict_batch(nets)
        netres_data = PosesData.from_list(net_results[0].pose_list)

        # 计算 ATE
        evaluator = Evaluation(ud.gt_data)
        evaluator.get_eval(netres_data, "netres")
        evaluator.print()
        evaluator.save(ud.base_dir / "eval.json")

    if dap.unit:
        ud = UnitData(dap.unit)
        action(ud)
    elif dap.dataset:
        dataset_path = dap.dataset
        datas = DeviceDataset(dataset_path)
        for ud in datas:
            action(ud)


if __name__ == "__main__":
    main()
