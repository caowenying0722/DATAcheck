from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt


def plot_one_cdf(
    cdf_data: dict,
    save_path: Path | str | None = None,
    show: bool = True,
):
    tag = cdf_data["tag"]
    errors = cdf_data["errors"]
    cdf = cdf_data["cdf"]
    percentiles = cdf_data["percentiles"]
    markers = ["o", "s", "^", "d"]

    label = tag
    plt.plot(
        errors,
        cdf,
        label=label,
        linewidth=2,
    )

    # 标记关键百分位点
    for pct, marker in zip(["90%", "95%", "99%"], markers):
        val = percentiles[pct]
        # 找到最接近的CDF值
        idx_closest = np.searchsorted(errors, val)
        if idx_closest < len(cdf):
            plt.plot(
                val,
                cdf[idx_closest],
                marker,
                markersize=8,
            )
            plt.annotate(
                f"{pct}={val:.4f}",
                xy=(val, cdf[idx_closest]),
                xytext=(5, -15),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
            )

    plt.xlabel("Error", fontsize=12)
    plt.ylabel("CDF", fontsize=12)
    plt.title(f"CDF of {tag}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # 设置y轴范围
    plt.ylim(0, 1.05)
    # 设置最大误差为 1，超出部分不显示
    plt.xlim(left=0, right=1)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"CDF plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_eval_cdf(
    inner: dict,
    tag: str = "",
    error_types: list[Literal["APE", "ATE", "RPE", "RTE"]]
    | Literal["APE", "ATE", "RPE", "RTE"] = "ATE",
    save_path: Path | str | None = None,
    show: bool = True,
):
    """
    绘制CDF曲线

    Args:
        eva_poses: 待评估的姿态数据
        tag: 结果标签
        error_types: 误差类型，可以是单个类型或类型列表
        save_path: 保存图片的路径，None则不保存
        show: 是否显示图片
    """
    if isinstance(error_types, str):
        error_types = [error_types]

    plt.figure(figsize=(10, 6))

    colors = ["blue", "red", "green", "orange"]
    markers = ["o", "s", "^", "d"]

    for idx, error_type in enumerate(error_types):
        cdf_data = inner[f"{error_type}_CDF"]
        errors = cdf_data["errors"]
        cdf = cdf_data["cdf"]
        percentiles = cdf_data["percentiles"]

        label = f"{error_type}"
        plt.plot(
            errors,
            cdf,
            label=label,
            color=colors[idx % len(colors)],
            linewidth=2,
        )

        # 标记关键百分位点
        for pct, marker in zip(["90%", "95%", "99%"], markers):
            val = percentiles[pct]
            # 找到最接近的CDF值
            idx_closest = np.searchsorted(errors, val)
            if idx_closest < len(cdf):
                plt.plot(
                    val,
                    cdf[idx_closest],
                    marker,
                    color=colors[idx % len(colors)],
                    markersize=8,
                )
                plt.annotate(
                    f"{pct}={val:.4f}",
                    xy=(val, cdf[idx_closest]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
                )

    plt.xlabel("Error (m or rad)", fontsize=12)
    plt.ylabel("CDF", fontsize=12)
    plt.title(f"CDF of {tag}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # 设置y轴范围
    plt.ylim(0, 1.05)
    plt.xlim(left=0)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"CDF plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
