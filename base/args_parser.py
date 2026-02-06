"""
NOTE:
    参数解析

Author: qi
"""

import argparse


class DatasetArgsParser:
    dataset: str | None = None
    group: str | None = None
    unit: str | None = None
    output: str | None = None

    # NOTE  Not default parameters
    regen: bool = False
    visual: bool = False
    time_range: tuple[float | None, float | None] = (None, None)

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="数据集参数解析")
        self.parser.add_argument("-d", "--dataset", type=str, help="数据集路径")
        self.parser.add_argument("-g", "--group", type=str, help="数据组路径")
        self.parser.add_argument("-u", "--unit", type=str, help="数据单元路径")
        self.parser.add_argument("-o", "--output", type=str, help="输出路径")
        self.parser.add_argument("-r", "--regen", action="store_true", help="重新生成")
        self.parser.add_argument("-v", "--visual", action="store_true", help="可视化")
        self.parser.add_argument(
            "-t", "--time_range", type=float, default=(None, None), nargs=2
        )

    def parse(self):
        self.parser.parse_args()
        self.args = self.parser.parse_args()

        print("命令行参数:")
        for k, v in self.args.__dict__.items():
            print(f"- {k}: {v}")
            setattr(self, k, v)

        return self
