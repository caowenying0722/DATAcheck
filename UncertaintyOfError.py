#!/usr/bin/env python3
"""
分析数据集成分


"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

import base.rerun_ext as bre
from base.args_parser import DatasetArgsParser
from base.datatype import DeviceDataset, UnitData
from base.evaluate import Evaluation
from base.model import DataRunner, InertialNetworkData, ModelLoader, NetworkResult
from base.obj import Obj

# 默认结果输出路径
EvalDir = Path("/Users/qi/Resources/results")


class NetworkResultAnalysis:
    def __init__(self, nr: NetworkResult):  # noqa: F821
        self.nr = nr
        self.name = self.nr.tag

    def analyze_uc_err(self):
        """
        Analyze the relationship between covariance and error for 3-axis data

        meas_cov_list: Model predicted covariance (uncertainty estimation) - 3x3 covariance matrices
        meas_list: Model predictions - 3D vectors (x, y, z)
        gt_list: Ground truth displacements - 3D vectors (x, y, z)
        """
        # Extract 3-axis data
        measurements = np.array(self.nr.meas_list)  # (N, 3)
        ground_truth = np.array(self.nr.gt_list)  # (N, 3)
        errors = measurements - ground_truth  # (N, 3) - error for each axis

        # Extract covariance matrices (diagonal elements for each axis)
        meas_cov = np.array(self.nr.meas_cov_list)
        meas_cov = np.exp(meas_cov)

        cov_x = meas_cov[:, 0]
        cov_y = meas_cov[:, 1]
        cov_z = meas_cov[:, 2]

        # Extract errors for each axis
        err_x = errors[:, 0]  # Error in x
        err_y = errors[:, 1]  # Error in y
        err_z = errors[:, 2]  # Error in z

        # Create figure with 3 subplots (one for each axis)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{self.name} 3-Axis Covariance vs Error Analysis", fontsize=16)

        axis_labels = ["X Axis", "Y Axis", "Z Axis"]
        cov_data = [cov_x, cov_y, cov_z]
        err_data = [err_x, err_y, err_z]
        colors = ["red", "green", "blue"]

        for i, (ax_label, cov, err, color) in enumerate(
            zip(axis_labels, cov_data, err_data, colors)
        ):
            # Scatter plot: Error vs Covariance
            ax = axes[i]
            ax.scatter(err, cov, alpha=0.5, s=20, color=color)
            ax.set_xlabel(f"{ax_label} Error")
            ax.set_ylabel(f"{ax_label} Covariance (Variance)")
            ax.set_title(f"{ax_label}: Error vs Covariance")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-0.01, 0.31)
            ax.grid(True, alpha=0.3)

            # Calculate correlation
            if len(cov) > 2:
                corr, p_value = pearsonr(err, cov)
                ax.text(
                    0.05,
                    0.95,
                    f"Correlation: {corr:.3f}\np-value: {p_value:.2e}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

        plt.tight_layout()

        # Save figure
        output_path = Path("covariance_error_analysis_3axis.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Analysis chart saved to: {output_path}")

        # Print statistics
        print("\n=== 3-Axis Covariance vs Error Analysis ===")
        print(f"Data points: {len(errors)}")

        for i, (ax_label, cov, err) in enumerate(zip(axis_labels, cov_data, err_data)):
            print(f"\n{ax_label} Statistics:")
            print(f"  Error - Mean: {np.mean(err):.6f}, Std: {np.std(err):.6f}")
            print(f"  Error - Min: {np.min(err):.6f}, Max: {np.max(err):.6f}")
            print(f"  Error - Median: {np.median(err):.6f}")
            print(f"  Covariance - Mean: {np.mean(cov):.6f}, Std: {np.std(cov):.6f}")
            print(f"  Covariance - Min: {np.min(cov):.6f}, Max: {np.max(cov):.6f}")
            print(f"  Covariance - Median: {np.median(cov):.6f}")

            if len(cov) > 2:
                corr, p_value = pearsonr(err, cov)
                print(f"  Correlation: {corr:.6f}, p-value: {p_value:.2e}")

                if abs(corr) > 0.7:
                    strength = "Strong"
                elif abs(corr) > 0.4:
                    strength = "Moderate"
                elif abs(corr) > 0.2:
                    strength = "Weak"
                else:
                    strength = "Very weak or none"

                direction = "Positive" if corr > 0 else "Negative"
                print(f"  Strength: {strength} {direction} correlation")

        plt.show()


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

    def action(ud: UnitData, res_dir: Path):
        print(f"> Eval {ud.name}")
        # 数据保存路径
        unit_out_dir = res_dir / ud.name
        unit_out_dir.mkdir(parents=True, exist_ok=True)

        obj_path = res_dir / "temp" / f"action_{ud.name}.pkl"
        print(f"> 存储路径：{obj_path}")
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        # 如果已经计算过
        if obj_path.exists():
            print(f"> 已存在结果：{obj_path}")
            nr_list, evaluator = Obj.load(obj_path)
            assert isinstance(nr_list, list)
            assert isinstance(evaluator, Evaluation)
        else:
            # 加载数据
            ud.load_data(using_opt=True)
            # 可视化
            bre.RerunView().add_spatial_view().send(ud.name)
            bre.send_pose_data(ud.gt_data, "Groundtruth", color=[192, 72, 72])

            # 模型推理
            nr_list = DataRunner(ud, Data, has_init_rerun=True).predict_batch(nets)

            # 计算 ATE
            evaluator = Evaluation(ud.gt_data, name=ud.name, rel_duration=1)
            evaluator.get_eval(nr_list[0].poses, f"{nets[0].name}_{ud.name}")
            evaluator.print()

            # 保存结果
            Obj.save((nr_list, evaluator), obj_path)

        nr = nr_list[0]
        # 分析
        # nra = NetworkResultAnalysis(nr)
        # nra.analyze_uc_err()
        return nr

    if dap.unit:
        unit_path = Path(dap.unit)
        # 使用 网络名称
        res_dir = EvalDir / f"{nets[0].name}"
        res_dir.mkdir(parents=True, exist_ok=True)

        ud = UnitData(unit_path)
        action(ud, res_dir)

    elif dap.dataset:
        dataset_path = Path(dap.dataset)
        datas = DeviceDataset(dataset_path)
        # 使用 网络名称 + 设备名称
        res_dir = EvalDir / f"{nets[0].name}_{datas.device_name}"
        res_dir.mkdir(parents=True, exist_ok=True)
        # 存储结果
        # da_path = res_dir / f"{DatasetAnalysis._obj_name}.pkl"

        all_nr = NetworkResult()

        for ud in datas:
            nr = action(ud, res_dir)
            all_nr.meas_cov_list.extend(nr.meas_cov_list)
            all_nr.gt_list.extend(nr.gt_list)
            all_nr.meas_list.extend(nr.meas_list)

        nra = NetworkResultAnalysis(all_nr)
        nra.analyze_uc_err()


if __name__ == "__main__":
    main()
