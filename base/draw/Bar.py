from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from . import Base2D


class Bar(Base2D):
    def __init__(
        self,
        y: NDArray | list[float],
        x_label: str,
        y_label: str,
        x: NDArray | list[float] | None = None,
        title: str = "",
    ):
        if x is None:
            x = np.arange(len(y))
        else:
            x = np.array(x)

        y = np.array(y)

        super().__init__(x, y, x_label, y_label, title)

    @staticmethod
    def from_dict(data: dict):
        return Bar(
            x=data["x"],
            y=data["y"],
            x_label=data["x_label"],
            y_label=data["y_label"],
            title=data["title"],
        )

    def draw(
        self,
        save_dir: Path | None = None,
        show: bool = True,
    ):
        """绘制柱状图,并标注平均值和中位数的虚线"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制柱状图
        ax.bar(self.x, self.y, alpha=0.7, label="Data")

        # 计算平均值和中位数
        mean_val = float(np.mean(self.y))
        median_val = float(np.median(self.y))

        # 绘制平均值虚线
        ax.axhline(
            y=mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.4f}",
        )

        # 绘制中位数虚线
        ax.axhline(
            y=median_val,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_val:.4f}",
        )

        # 标记最大值
        max_idx = int(np.argmax(self.y))
        max_val = float(self.y[max_idx])
        max_x = self.x[max_idx]
        ax.annotate(
            f"Max: {max_val:.4f}",
            xy=(max_x, max_val),
            xytext=(max_x, max_val + (max_val - median_val) * 0.1),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            fontsize=10,
            ha="center",
        )

        # 设置标签和标题
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_title(self.title)
        ax.legend()

        return fig
