from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from base.obj import ObjSaver


class UnitAnalysis(ObjSaver):
    """
    数据集分析类，用于分析数据集的分布情况

    分析数据集中的速度构成，转弯构成。
    """

    def __init__(
        self,
        tag: str,
        vels: list[NDArray] | NDArray,
        rate: int = 20,
    ):
        self.tag = tag
        self.velocities = vels
        self.rate = rate

        self.turns = {}

    def analyze(self, save_dir: Path):
        obj_path = save_dir / self._obj_name
        turn_png_path = save_dir / "turns_analysis.png"

        # 角度分析
        self.turns = self.calculate_turns(self.velocities, self.rate)
        self.save(obj_path)

        self.plot_turns_and_trajectory(self.turns, save_path=turn_png_path, show=False)

    @staticmethod
    def calculate_turns(
        velocities: list[NDArray] | NDArray,
        window_size: int = 20,
        threshold: float = 1.5,
        rate: int = 20,
    ):
        """
        给定速度序列，计算数据集中的转弯构成。

        Args:
            velocities: 速度数组 (N, 3) 或 (N, 2)，单位 m/s，频率 rate Hz
            window_size: 平滑窗口大小
            threshold: 转弯角度阈值（度）
            rate: 数据采样频率 (Hz)，默认20Hz

        Returns:
            包含转弯分析结果的字典
        """
        vel = np.array(velocities)
        n = len(vel)
        dt = 1.0 / rate

        # 1. 从速度积分得到位置
        positions = np.cumsum(vel * dt, axis=0)

        # 2. 计算速度大小
        speed = np.linalg.norm(vel[:, :2], axis=1)

        # 3. 计算航向角
        headings_raw = np.rad2deg(np.arctan2(vel[:, 1], vel[:, 0]))

        # 4. 展开角度：消除 ±180° 突变
        headings_unwrapped = np.zeros_like(headings_raw)
        headings_unwrapped[0] = headings_raw[0]
        for i in range(1, n):
            diff = headings_raw[i] - headings_unwrapped[i - 1]
            # 如果跳变超过 180°，说明跨越了 ±180° 边界，需要补偿
            if diff > 180:
                headings_raw[i] -= 360
            elif diff < -180:
                headings_raw[i] += 360
            headings_unwrapped[i] = headings_raw[i]

        # 5. 使用移动平均平滑航向角
        half_win = window_size // 2
        headings_smooth = np.zeros(n)
        for i in range(n):
            start = max(0, i - half_win)
            end = min(n, i + half_win + 1)
            # 对展开后的角度直接平均
            headings_smooth[i] = np.mean(headings_unwrapped[start:end])

        # 6. 计算角度变化
        angle_diff = np.diff(headings_smooth)
        angle_changes = np.abs(angle_diff)

        # 7. 检测转弯点
        turn_indices = np.where(angle_changes > threshold)[0]

        # 8. 计算轨迹长度和转弯轨迹长度
        # 计算相邻点之间的距离
        displacements = np.diff(positions[:, :2], axis=0)
        segment_lengths = np.linalg.norm(displacements, axis=1)
        total_length = np.sum(segment_lengths)

        # 计算转弯段的长度（转弯点及其后续段）
        turn_length = 0.0
        if len(turn_indices) > 0:
            # 对每个转弯点，计算该点到下一个点的距离
            turn_length = np.sum(segment_lengths[turn_indices])

        return {
            "window_size": window_size,
            "threshold": threshold,
            "rate": rate,
            "turn_count": len(turn_indices),
            "turn_ratio": len(turn_indices) / n,
            "turn_indices": turn_indices,
            "angle_changes": angle_changes,
            "mean_turn_angle": np.mean(angle_changes[turn_indices])
            if len(turn_indices) > 0
            else 0,
            "positions": positions,
            "headings": headings_unwrapped,  # 展开后的航向角
            "headings_smooth": headings_smooth,
            "speed": speed,
            "total_length": total_length,  # 轨迹总长度
            "turn_length": turn_length,  # 转弯段长度
            "turn_length_ratio": turn_length / total_length
            if total_length > 0
            else 0,  # 转弯长度占比
        }

    @staticmethod
    def plot_turns_and_trajectory(
        turns_result: dict,
        save_path: Path | None = None,
        show: bool = True,
    ):
        """
        绘制转弯分析和轨迹关系的可视化图

        Args:
            turns_result: calculate_turns 返回的结果字典
            save_path: 保存图像路径
            show: 是否显示图像
        """
        # 从 turns_result 读取所有数据和参数
        positions = turns_result["positions"]
        speed = turns_result["speed"]
        headings_smooth = turns_result["headings_smooth"]
        angle_changes = turns_result["angle_changes"]
        turn_indices = turns_result["turn_indices"]
        threshold = turns_result["threshold"]
        window_size = turns_result["window_size"]
        rate = turns_result["rate"]

        # 只取 x, y 平面
        if positions.shape[1] == 3:
            positions_2d = positions[:, :2]
        else:
            positions_2d = positions

        # 创建图形 - 2x2布局
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])

        # 1. 轨迹图（上方，占两列）
        ax_traj = fig.add_subplot(gs[0, :])

        # 根据转角状态给轨迹着色
        turn_mask = np.zeros(len(positions_2d), dtype=bool)
        turn_mask[turn_indices] = True

        # 绘制正常段轨迹（蓝色）
        normal_traj = positions_2d[~turn_mask]
        if len(normal_traj) > 0:
            ax_traj.plot(
                normal_traj[:, 0],
                normal_traj[:, 1],
                "b.",
                linewidth=0.5,
                alpha=0.3,
                markersize=2,
                label="Normal Motion",
            )

        # 绘制转弯点（红色，更大）
        if len(turn_indices) > 0:
            ax_traj.scatter(
                positions_2d[turn_indices, 0],
                positions_2d[turn_indices, 1],
                c="red",
                s=30,
                marker="o",
                label=f"Turn Points ({len(turn_indices)})",
                zorder=5,
                edgecolors="darkred",
                linewidths=0.5,
                alpha=0.4,
            )

            # 标记起点和终点
            ax_traj.scatter(
                positions_2d[0, 0],
                positions_2d[0, 1],
                c="green",
                s=50,
                marker="^",
                label="Start",
                zorder=6,
                edgecolors="darkgreen",
                linewidths=2,
            )
            ax_traj.scatter(
                positions_2d[-1, 0],
                positions_2d[-1, 1],
                c="orange",
                s=50,
                marker="s",
                label="End",
                zorder=6,
                edgecolors="darkorange",
                linewidths=2,
            )

        ax_traj.set_xlabel("X Position (m)", fontsize=12)
        ax_traj.set_ylabel("Y Position (m)", fontsize=12)
        ax_traj.set_title(
            "Trajectory with Turn Detection", fontsize=14, fontweight="bold"
        )
        ax_traj.legend(loc="best")
        ax_traj.grid(True, alpha=0.3)
        ax_traj.axis("equal")

        # 2. 速度曲线（中左）
        ax_speed = fig.add_subplot(gs[1, 0])
        time_axis = np.arange(len(speed)) / rate
        ax_speed.plot(time_axis, speed, "b-", linewidth=0.8, alpha=0.7, label="Speed")

        # 高亮转角点对应的速度
        if len(turn_indices) > 0:
            ax_speed.scatter(
                time_axis[turn_indices],
                speed[turn_indices],
                c="red",
                s=15,
                marker="o",
                alpha=0.4,
                zorder=5,
                label="Turn Points",
            )

        ax_speed.set_xlabel("Time (s)", fontsize=11)
        ax_speed.set_ylabel("Speed (m/s)", fontsize=11)
        ax_speed.set_title("Speed Profile", fontsize=12, fontweight="bold")
        ax_speed.legend(fontsize=9)
        ax_speed.grid(True, alpha=0.3)

        # 3. 航向角曲线（中右）
        ax_heading = fig.add_subplot(gs[1, 1])
        ax_heading.plot(
            time_axis,
            headings_smooth,
            "g-",
            linewidth=0.8,
            alpha=0.7,
            label="Heading (smooth)",
        )

        # 高亮转角点
        if len(turn_indices) > 0:
            ax_heading.scatter(
                time_axis[turn_indices],
                headings_smooth[turn_indices],
                c="red",
                s=15,
                marker="o",
                alpha=0.4,
                zorder=5,
            )

        ax_heading.set_xlabel("Time (s)", fontsize=11)
        ax_heading.set_ylabel("Heading (°)", fontsize=11)
        ax_heading.set_title("Heading Angle", fontsize=12, fontweight="bold")
        ax_heading.legend(fontsize=9)
        ax_heading.grid(True, alpha=0.3)

        # 4. 角度变化图（下左）
        ax_angle = fig.add_subplot(gs[2, 0])
        angle_time = np.arange(len(angle_changes)) / rate

        # 绘制角度变化曲线
        ax_angle.plot(
            angle_time,
            angle_changes,
            "g-",
            linewidth=0.8,
            alpha=0.7,
            label="Angle Change",
        )

        # 绘制阈值线
        ax_angle.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({threshold}°)",
        )

        # 高亮转弯区域
        if len(turn_indices) > 0:
            ax_angle.scatter(
                angle_time[turn_indices],
                angle_changes[turn_indices],
                c="red",
                s=15,
                marker="o",
                alpha=0.4,
                zorder=5,
                label="Detected Turns",
            )

        ax_angle.set_xlabel("Time (s)", fontsize=11)
        ax_angle.set_ylabel("Angle Change (°)", fontsize=11)
        ax_angle.set_title("Heading Angle Changes", fontsize=12, fontweight="bold")
        ax_angle.legend(fontsize=9)
        ax_angle.grid(True, alpha=0.3)

        # 5. 统计信息文本框（下右）
        ax_stats = fig.add_subplot(gs[2, 1])
        ax_stats.axis("off")

        # 从结果中读取长度信息
        total_length = turns_result.get("total_length", 0)
        turn_length = turns_result.get("turn_length", 0)
        turn_length_ratio = turns_result.get("turn_length_ratio", 0)

        # 统计信息
        stats_text = f"""
Turn Statistics Summary
{"=" * 35}
Total Samples: {len(positions)}
Sample Rate: {rate} Hz
Duration: {len(positions) / rate:.2f} s
Window Size: {window_size}
Threshold: {threshold}°
Turn Count: {turns_result["turn_count"]}
Turn Ratio: {turns_result["turn_ratio"]:.2%}
Mean Turn Angle: {turns_result["mean_turn_angle"]:.2f}°
Max Angle Change: {np.max(angle_changes):.2f}°
Std Angle Change: {np.std(angle_changes):.2f}°

Trajectory Info:
{"=" * 35}
Total Length: {total_length:.2f} m
Turn Length: {turn_length:.2f} m
Turn Length Ratio: {turn_length_ratio:.2%}
Mean Speed: {np.mean(speed):.3f} m/s
Max Speed: {np.max(speed):.3f} m/s
Min Speed: {np.min(speed):.3f} m/s
X Range: [{np.min(positions_2d[:, 0]):.2f}, {np.max(positions_2d[:, 0]):.2f}] m
Y Range: [{np.min(positions_2d[:, 1]):.2f}, {np.max(positions_2d[:, 1]):.2f}] m
        """

        ax_stats.text(
            0.1,
            0.5,
            stats_text,
            transform=ax_stats.transAxes,
            fontsize=9,
            verticalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"图像已保存至: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig


class DatasetAnalysis(ObjSaver):
    ua_list: list[UnitAnalysis]

    def __init__(self):
        self.ua_list = []

    def add(self, ua: UnitAnalysis):
        self.ua_list.append(ua)

    def analyze(self, save_dir: Path, tag: str):
        """
        汇总分析所有单元的转弯和速度分布

        Args:
            save_dir: 保存目录
            device_name: 设备名称
        """
        if len(self.ua_list) == 0:
            print("No unit analysis results to summarize.")
            return

        # 1. 收集所有单元的转弯分析结果和速度
        turns_result_list = [ua.turns for ua in self.ua_list]
        all_speeds = [ua.turns["speed"] for ua in self.ua_list]

        # 2. 统计转弯占比
        total_length = sum(r["total_length"] for r in turns_result_list)
        total_turn_length = sum(r["turn_length"] for r in turns_result_list)
        overall_turn_ratio = total_turn_length / total_length if total_length > 0 else 0

        # 3. 汇总转弯统计
        total_turns = sum(r["turn_count"] for r in turns_result_list)
        mean_turn_ratio = np.mean([r["turn_ratio"] for r in turns_result_list])
        mean_turn_angle = np.mean([r["mean_turn_angle"] for r in turns_result_list])

        # 4. 速度分布统计
        all_speeds_flat = np.concatenate(all_speeds)
        speed_mean = np.mean(all_speeds_flat)
        speed_std = np.std(all_speeds_flat)
        speed_median = np.median(all_speeds_flat)
        speed_min = np.min(all_speeds_flat)
        speed_max = np.max(all_speeds_flat)

        # 5. 创建汇总图表
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

        # 5.1 转弯占比饼图
        ax_pie = fig.add_subplot(gs[0, 0])
        labels = [
            f"Turn\n({overall_turn_ratio:.1%})",
            f"Straight\n({1 - overall_turn_ratio:.1%})",
        ]
        colors = ["#ff6b6b", "#4ecdc4"]
        sizes = [overall_turn_ratio, 1 - overall_turn_ratio]
        ax_pie.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 12},
        )
        ax_pie.set_title("Overall Turn Length Ratio", fontsize=14, fontweight="bold")

        # 5.2 转弯统计条形图
        ax_bar = fig.add_subplot(gs[0, 1])
        unit_names = [f"{ua.tag}" for ua in self.ua_list]
        turn_ratios = [r["turn_length_ratio"] for r in turns_result_list]

        ax_bar.bar(unit_names, turn_ratios, color="#ff6b6b", alpha=0.7)
        ax_bar.axhline(
            y=overall_turn_ratio,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {overall_turn_ratio:.1%}",
        )
        ax_bar.set_xlabel("Unit", fontsize=12)
        ax_bar.set_ylabel("Turn Length Ratio", fontsize=12)
        ax_bar.set_title("Turn Ratio by Unit", fontsize=14, fontweight="bold")
        ax_bar.legend()
        ax_bar.grid(True, alpha=0.3, axis="y")
        # ax_bar.set_ylim([0, max(max(turn_ratios) * 1.2, overall_turn_ratio * 1.2)])

        # 旋转 x 轴标签，防止遮挡
        plt.setp(ax_bar.get_xticklabels(), rotation=45, ha="right")

        # 5.3 速度分布直方图
        ax_hist = fig.add_subplot(gs[1, 0])
        ax_hist.hist(
            all_speeds_flat,
            bins=50,
            color="#4ecdc4",
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        ax_hist.axvline(
            speed_mean,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {speed_mean:.3f} m/s",
        )
        ax_hist.axvline(
            speed_median,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Median: {speed_median:.3f} m/s",
        )
        ax_hist.set_xlabel("Speed (m/s)", fontsize=12)
        ax_hist.set_ylabel("Frequency", fontsize=12)
        ax_hist.set_title("Speed Distribution", fontsize=14, fontweight="bold")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3, axis="y")

        # 5.4 统计信息文本框
        ax_stats = fig.add_subplot(gs[1, 1])
        ax_stats.axis("off")

        stats_text = f"""
Dataset Summary: {tag}
{"=" * 40}

Turn Statistics:
  Total Units: {len(turns_result_list)}
  Total Length: {total_length:.2f} m
  Total Turn Length: {total_turn_length:.2f} m
  Overall Turn Ratio: {overall_turn_ratio:.2%}
  Mean Turn Ratio: {mean_turn_ratio:.2%}
  Total Turns: {total_turns}
  Mean Turn Angle: {mean_turn_angle:.2f}°

Speed Statistics:
  Mean Speed: {speed_mean:.3f} m/s
  Median Speed: {speed_median:.3f} m/s
  Std Speed: {speed_std:.3f} m/s
  Min Speed: {speed_min:.3f} m/s
  Max Speed: {speed_max:.3f} m/s
  Total Samples: {len(all_speeds_flat)}
        """

        ax_stats.text(
            0.1,
            0.5,
            stats_text,
            transform=ax_stats.transAxes,
            fontsize=11,
            verticalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.tight_layout()

        # 保存图表
        save_path = save_dir / f"{tag}_summary.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"汇总统计图已保存至: {save_path}")
        plt.close()

        # 打印统计信息
        print("\n" + "=" * 60)
        print(f"Dataset Summary: {tag}")
        print("=" * 60)
        print(f"Total Units: {len(turns_result_list)}")
        print(f"Total Length: {total_length:.2f} m")
        print(f"Total Turn Length: {total_turn_length:.2f} m")
        print(f"Overall Turn Ratio: {overall_turn_ratio:.2%}")
        print(f"Mean Turn Angle: {mean_turn_angle:.2f}°")
        print(f"Mean Speed: {speed_mean:.3f} ± {speed_std:.3f} m/s")
        print("=" * 60 + "\n")
