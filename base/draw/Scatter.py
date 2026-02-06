import matplotlib.pyplot as plt
from scipy import stats

from . import Base2D


class Scatter(Base2D):
    def _draw_inner(self):
        """
        绘制散点图并显示相关性指标
        """
        # 计算相关性指标
        pearson_corr, pearson_p = stats.pearsonr(self.x, self.y)
        spearman_corr, spearman_p = stats.spearmanr(self.x, self.y)
        kendall_corr, kendall_p = stats.kendalltau(self.x, self.y)

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制散点图
        _scatter = ax.scatter(
            self.x, self.y, alpha=0.6, s=10, edgecolors="k", linewidths=0.1
        )

        # 设置标签和标题
        ax.set_xlabel(self.x_label, fontsize=12)
        ax.set_ylabel(self.y_label, fontsize=12)
        ax.set_title(self.title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 在图上显示相关性指标
        stats_text = "Correlation Metrics:\n"
        stats_text += f"Pearson r: {pearson_corr:.4f} (p={pearson_p:.2e})\n"
        stats_text += f"Spearman ρ: {spearman_corr:.4f} (p={spearman_p:.2e})\n"
        stats_text += f"Kendall τ: {kendall_corr:.4f} (p={kendall_p:.2e})"

        # 将文本框放在图的左上角
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        return fig
