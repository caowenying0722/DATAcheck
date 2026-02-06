#!/usr/bin/env python3
""" """

from base.args_parser import DatasetArgsParser
from base.datatype import DeviceDataset, UnitData
from base.model import DataRunner, InertialNetworkData, ModelLoader


def main():
    dap = DatasetArgsParser()
    dap.parser.add_argument("-m", "--models", nargs="+", help="模型")
    dap.parser.add_argument("--models_path", type=str, help="模型文件夹")
    dap.parse()

    models = dap.args.models
    if models is None or len(models) == 0:
        models = ["model_mi_hw_1129"]

    models_path = "models"
    if dap.args.models_path is not None:
        models_path = dap.args.models_path
    loader = ModelLoader(models_path)

    Data = InertialNetworkData.set_step(20)
    if dap.unit:
        # 数据
        data = UnitData(dap.unit)
        runner = DataRunner(data, Data)
        runner.predict_batch(loader.get_by_names(models))

    elif dap.dataset:
        dataset_path = dap.dataset
        datas = DeviceDataset(dataset_path)
        for data in datas:
            runner = DataRunner(data, Data)
            runner.predict_batch(loader.get_by_names(models))
    else:
        dap.parser.print_help()


# if __name__ == "__main__":
# main()
