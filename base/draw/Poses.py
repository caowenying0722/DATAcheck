from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def draw_trajectory_2d(
    poses_data,
    ax=None,
    title: str = "Trajectory",
    xlabel: str = "X (m)",
    ylabel: str = "Y (m)",
    show: bool = True,
    linewidth: float = 1.5,
    color: str = "blue",
    start_marker_size: float = 10,
    end_marker_size: float = 10,
    figsize: tuple[float, float] = (10, 10),
    save_path: str | None = None,
    dpi: int = 300,
):
    """
    绘制PosesData类型的xy二维轨迹图，xy轴尺度一致，出发点和结束点用圆点和叉标记。

    Args:
        poses_data: PosesData对象，包含轨迹数据
        ax: matplotlib的Axes对象，如果为None则创建新的figure和axes
        title: 图像标题
        xlabel: x轴标签
        ylabel: y轴标签
        show: 是否显示图像
        linewidth: 轨迹线宽
        color: 轨迹颜色
        start_marker_size: 起点标记大小
        end_marker_size: 终点标记大小
        figsize: 图像大小
        save_path: 保存图像的路径，如果为None则不保存
        dpi: 保存图像的分辨率

    Returns:
        ax: matplotlib的Axes对象
    """
    # 提取位置数据（假设poses_data.ps是N x 3的数组）
    positions = poses_data.ps
    x = positions[:, 0]
    y = positions[:, 1]

    # 创建新的figure和axes，如果未提供
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # 绘制轨迹线
    ax.plot(x, y, color=color, linewidth=linewidth, label="Trajectory")

    # 绘制起点（用圆点）
    ax.plot(
        x[0],
        y[0],
        marker="o",
        color="green",
        markersize=start_marker_size,
        label="Start",
        linestyle="None",
    )

    # 绘制终点（用叉）
    ax.plot(
        x[-1],
        y[-1],
        marker="x",
        color="red",
        markersize=end_marker_size,
        label="End",
        linestyle="None",
    )

    # 设置标题和标签
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 确保xy轴尺度一致
    ax.set_aspect("equal")

    plt.tight_layout()

    # 保存图像
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = ax.figure
        assert isinstance(fig, Figure)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to: {path}")

    # 显示图像
    if show:
        plt.show()

    return ax


def draw_trajectory_2d_compare(
    poses_data_list,
    labels: list[str],
    colors: list[str] | None = None,
    ax=None,
    title: str = "Trajectory Comparison",
    xlabel: str = "X (m)",
    ylabel: str = "Y (m)",
    show: bool = True,
    linewidth: float = 1.5,
    start_marker_size: float = 10,
    end_marker_size: float = 10,
    figsize: tuple[float, float] = (10, 10),
    save_path: str | None = None,
    dpi: int = 300,
):
    """
    绘制多个PosesData类型的xy二维轨迹图进行对比，xy轴尺度一致。

    Args:
        poses_data_list: PosesData对象列表
        labels: 每条轨迹的标签
        colors: 每条轨迹的颜色列表，如果为None则自动生成
        ax: matplotlib的Axes对象，如果为None则创建新的figure和axes
        title: 图像标题
        xlabel: x轴标签
        ylabel: y轴标签
        show: 是否显示图像
        linewidth: 轨迹线宽
        start_marker_size: 起点标记大小
        end_marker_size: 终点标记大小
        figsize: 图像大小
        save_path: 保存图像的路径，如果为None则不保存
        dpi: 保存图像的分辨率

    Returns:
        ax: matplotlib的Axes对象
    """
    n_trajectories = len(poses_data_list)

    if len(labels) != n_trajectories:
        raise ValueError(
            f"Number of labels ({len(labels)}) must match number of trajectories ({n_trajectories})"
        )

    # 自动生成颜色
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_trajectories))  # pyright: ignore[reportAttributeAccessIssue]

    # 创建新的figure和axes，如果未提供
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # 绘制每条轨迹
    for i, poses_data in enumerate(poses_data_list):
        positions = poses_data.ps
        x = positions[:, 0]
        y = positions[:, 1]

        color = colors[i]  # pyright: ignore[reportOptionalSubscript]

        # 绘制轨迹线
        ax.plot(x, y, color=color, linewidth=linewidth, label=labels[i], alpha=0.7)

        # 绘制起点（用圆点）
        ax.plot(
            x[0],
            y[0],
            marker="o",
            color=color,
            markersize=start_marker_size,
            linestyle="None",
        )

        # 绘制终点（用叉）
        ax.plot(
            x[-1],
            y[-1],
            marker="x",
            color=color,
            markersize=end_marker_size,
            linestyle="None",
        )

    # 设置标题和标签
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 确保xy轴尺度一致
    ax.set_aspect("equal")

    plt.tight_layout()

    # 保存图像
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = ax.figure
        assert isinstance(fig, Figure)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to: {path}")

    # 显示图像
    if show:
        plt.show()

    return ax
