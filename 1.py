"""
标准评估器 - 用于评估 IMU 数据与 Ground Truth 的对齐精度

Features:
- 多层级路径支持: dataset/group/unit
- 指标统一保存: CSV + JSON
- 可视化: 时间匹配图 + 轨迹图 (matplotlib + rerun)
- 时间范围选择: 支持按绝对时间范围评估数据
- 空间校准: Final 阶段可选的空间校准功能

Author: refactored by Claude


# 单个序列
uv run StandardEvaluator.py -u path/to/unit -v

# 单个序列，指定时间范围（第5秒到第15秒）
uv run StandardEvaluator.py -u path/to/unit -t 5.0 15.0 -v

# 单个序列，启用 Final 阶段的空间校准
uv run StandardEvaluator.py -u path/to/unit --enable-calibration -v

# 批量处理
uv run StandardEvaluator.py -d path/to/dataset -v

"""

import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from scipy.spatial.transform import Rotation
from scipy.signal import correlate, find_peaks
from scipy.fft import fft
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import rerun as rr

from base.datatype import ImuData, GroundTruthData, UnitData
from base.args_parser import DatasetArgsParser
from base.rerun_ext import RerunView, send_pose_data, send_imu_data


# ==================== 数据结构 ====================

@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    # 基础信息
    unit_name: str
    dataset: str = ""
    group: str = ""

    # Original指标（完全不应用g_rot旋转）
    original_rmse: float = 0.0
    original_corr: float = 0.0
    original_grav_err: float = 0.0
    original_grav_mag: float = 0.0

    # Raw指标（应用g_rot旋转）
    raw_rmse: float = 0.0
    raw_corr: float = 0.0
    raw_grav_err: float = 0.0
    raw_grav_mag: float = 0.0

    # 时间对齐后
    time_rmse: float = 0.0
    time_corr: float = 0.0
    time_grav_err: float = 0.0
    time_grav_mag: float = 0.0
    time_shift_ms: float = 0.0

    # 完整校准后
    final_rmse: float = 0.0
    final_corr: float = 0.0
    final_grav_err: float = 0.0
    final_grav_mag: float = 0.0
    calib_euler_x: float = 0.0
    calib_euler_y: float = 0.0
    calib_euler_z: float = 0.0

    # 时移敏感指标
    sign_consistency: float = 0.0
    peak_alignment: float = 0.0
    energy_corr: float = 0.0


# ==================== 参数解析器 ====================

class EvaluatorArgsParser:
    """评估器参数解析器"""

    def __init__(self):
        self.parser = DatasetArgsParser()
        # 扩展参数（避免与 DatasetArgsParser 冲突）
        self.parser.parser.add_argument(
            "--rerun", action="store_true",
            help="使用 rerun 进行 3D 可视化"
        )
        self.parser.parser.add_argument(
            "--z-axis-up", action="store_true", default=True,
            help="Z轴向上（默认：[0,0,1]）"
        )
        self.parser.parser.add_argument(
            "--z-axis-down", action="store_false", dest="z_axis_up",
            help="Z轴向下（[0,0,-1]）"
        )
        self.parser.parser.add_argument(
            "-R", "--recursive", action="store_true",
            help="递归搜索子目录中的数据"
        )
        self.parser.parser.add_argument(
            "--enable-calibration", action="store_true",
            help="启用 Final 阶段的空间校准（可选）"
        )

    def parse(self):
        """解析命令行参数"""
        self.parser.parse()
        return self.parser


# ==================== 核心评估器 ====================

class StandardEvaluator:
    """标准评估器 - 核心评估逻辑"""

    def __init__(self, unit: UnitData, dataset: str = "", group: str = "",
                 unit_path: Path = None, save_plots: bool = False, z_axis_up: bool = True,
                 time_range: tuple = None, enable_calibration: bool = False):
        """
        Args:
            unit: 单元数据
            dataset: 数据集名称
            group: 数据组名称
            unit_path: 序列路径（用于保存图表）
            save_plots: 是否保存可视化图表
            z_axis_up: Z轴方向（True=向上[0,0,1], False=向下[0,0,-1]）
            time_range: 时间范围（秒），(start_time, end_time)，None 表示使用全部数据
            enable_calibration: 是否启用 Final 阶段的空间校准
        """
        self.unit = unit
        self.name = unit.name
        self.dataset = dataset
        self.group = group
        self.unit_path = unit_path
        self.save_plots = save_plots
        self.imu = unit.imu_data
        self.gt = unit.gt_data
        # 重力方向：Z轴向上为正，向下为负
        self.z_axis = np.array([0, 0, 1 if z_axis_up else -1])
        self.time_range = time_range
        self.enable_calibration = enable_calibration

        # 计算 GT 角速度
        self._compute_gt_gyro()

        # 截取稳定段
        self._slice_stable_data()

    def _compute_gt_gyro(self):
        """计算 GT 角速度（通过姿态微分）"""
        dt = np.mean(np.diff(self.gt.t_us)) / 1e6
        if dt == 0:
            dt = 0.01

        R = self.gt.rots.as_matrix()
        R_diff = np.einsum('nij,njk->nik', R[:-1].transpose(0, 2, 1), R[1:])
        gt_w = Rotation.from_matrix(R_diff).as_rotvec() / dt
        self.gt_gyro_full = np.vstack([gt_w, np.zeros(3)])

    def _slice_stable_data(self, start_ratio: float = 0.1, end_ratio: float = 0.9):
        """截取稳定段数据

        Args:
            start_ratio: 起始比例（仅当 time_range 为 None 时使用）
            end_ratio: 结束比例（仅当 time_range 为 None 时使用）
        """
        # 确定时间范围（秒）
        if self.time_range is not None:
            # 使用绝对时间范围
            t_start_sec, t_end_sec = self.time_range
            # 转换为微秒
            t_start_us = t_start_sec * 1e6
            t_end_us = t_end_sec * 1e6
            # 基于时间戳选择数据范围
            time_mask = (self.imu.t_us >= t_start_us) & (self.imu.t_us <= t_end_us)
            indices = np.where(time_mask)[0]
            if len(indices) == 0:
                print(f"⚠️  警告: 时间范围 [{t_start_sec}, {t_end_sec}] 秒内无数据")
                print(f"   数据时间范围: [{self.imu.t_us[0]/1e6:.2f}, {self.imu.t_us[-1]/1e6:.2f}] 秒")
                # 回退到默认比例
                total = len(self.imu.gyro)
                start_idx = int(total * start_ratio)
                end_idx = int(total * end_ratio)
                s = slice(start_idx, end_idx)
            else:
                start_idx = indices[0]
                end_idx = indices[-1] + 1
                s = slice(start_idx, end_idx)
                print(f"✓ 使用时间范围: [{t_start_sec}, {t_end_sec}] 秒 (索引: {start_idx}-{end_idx})")
        else:
            # 使用比例范围（默认行为）
            total = len(self.imu.gyro)
            s = slice(int(total * start_ratio), int(total * end_ratio))

        self.i_gyro_eval = self.imu.gyro[s]
        self.i_acce_eval = self.imu.acce[s]
        self.i_t_eval = self.imu.t_us[s].astype(np.float64)

        self.g_gyro_eval_original = self.gt_gyro_full[s]
        # 按模长裁剪：确保向量模长不超过 10.0 rad/s（行人角速度合理范围）
        gt_norms = np.linalg.norm(self.g_gyro_eval_original, axis=1, keepdims=True)
        scale_factors = np.minimum(1.0, 10.0 / gt_norms)
        self.g_gyro_eval = self.g_gyro_eval_original * scale_factors
        self.g_rots_eval = self.gt.rots[s]
        self.g_t_eval = self.gt.t_us[s].astype(np.float64)

        # 诊断 GT 尖峰情况（如果需要保存图表）
        if self.save_plots and self.unit_path:
            self._diagnose_gt_spikes(save_to_dir=True, unit_path=self.unit_path)
        else:
            self._diagnose_gt_spikes(save_to_dir=False)

    def _diagnose_gt_spikes(self, save_to_dir: bool = False, unit_path: Path = None):
        """诊断 GT 角速度尖峰情况

        Args:
            save_to_dir: 是否保存可视化图表到序列目录
            unit_path: 序列路径（用于保存图表）
        """
        gt_norm_original = np.linalg.norm(self.g_gyro_eval_original, axis=1)
        gt_norm_clipped = np.linalg.norm(self.g_gyro_eval, axis=1)
        spike_count = np.sum(gt_norm_original > 10.0)
        spike_ratio = spike_count / len(gt_norm_original) * 100

        if spike_count > 0:
            print(f"\n⚠️  GT角速度尖峰检测:")
            print(f"  尖峰数量: {spike_count} / {len(gt_norm_original)} ({spike_ratio:.2f}%)")
            print(f"  超过阈值 (>10.0 rad/s) 的值已被 clip 处理")
            print(f"  原始最大值: {np.max(gt_norm_original):.2f} rad/s")
            print(f"  clip后最大值: {np.max(gt_norm_clipped):.2f} rad/s")
            print(f"  原始标准差: {np.std(gt_norm_original):.4f} rad/s")
            print(f"  clip后标准差: {np.std(gt_norm_clipped):.4f} rad/s")

            # 可视化尖峰情况
            if save_to_dir and unit_path:
                self._plot_gt_spikes(
                    self.name,
                    gt_norm_original,
                    gt_norm_clipped,
                    self.g_t_eval,
                    unit_path
                )

    def _plot_gt_spikes(self, name: str, gt_norm_original: np.ndarray,
                        gt_norm_clipped: np.ndarray, gt_t_us: np.ndarray, unit_path: Path):
        """绘制 GT 尖峰对比图

        Args:
            name: 序列名称
            gt_norm_original: 原始 GT 角速度模长
            gt_norm_clipped: clip 后的 GT 角速度模长
            gt_t_us: GT 时间戳（微秒）
            unit_path: 序列路径（保存目录）
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # 转换为秒
        t_sec = gt_t_us / 1e6
        t_sec = t_sec - t_sec[0]  # 从 0 开始

        # 子图1：原始 GT 角速度
        axes[0].plot(t_sec, gt_norm_original, 'r-', alpha=0.7, linewidth=0.5, label='Original GT Gyro')
        axes[0].axhline(10.0, color='orange', linestyle='--', linewidth=1.5, label='Clip Threshold (10.0 rad/s)')
        axes[0].set_ylabel('Angular Velocity (rad/s)')
        axes[0].set_title(f'GT Gyro Norm: ORIGINAL (with spikes) - {name}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(bottom=0)

        # 标记尖峰
        spike_indices = gt_norm_original > 10.0
        if np.sum(spike_indices) > 0:
            axes[0].scatter(t_sec[spike_indices], gt_norm_original[spike_indices],
                           c='red', s=10, alpha=0.5, zorder=5, label=f'Spikes ({np.sum(spike_indices)} points)')

        # 子图2：clip 后的 GT 角速度
        axes[1].plot(t_sec, gt_norm_clipped, 'g-', alpha=0.7, linewidth=0.8, label='Clipped GT Gyro')
        axes[1].axhline(10.0, color='orange', linestyle='--', linewidth=1.5, label='Clip Threshold')
        axes[1].set_ylabel('Angular Velocity (rad/s)')
        axes[1].set_title(f'GT Gyro Norm: CLIPPED (spikes removed) - {name}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(bottom=0)

        # 子图3：对比
        axes[2].plot(t_sec, gt_norm_original, 'r-', alpha=0.5, linewidth=1, label='Original')
        axes[2].plot(t_sec, gt_norm_clipped, 'g-', alpha=0.7, linewidth=1.5, label='Clipped')
        axes[2].axhline(10.0, color='orange', linestyle='--', linewidth=1, label='Threshold')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Angular Velocity (rad/s)')
        axes[2].set_title(f'Original vs Clipped Comparison - {name}')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        save_path = unit_path / "gt_spikes.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  📈 GT spikes plot saved: {save_path}")

    def compute_metrics(self, i_gyro, i_acce, g_gyro, g_rot) -> Dict[str, float]:
        """计算评估指标（应用g_rot旋转）"""
        diff = i_gyro - g_gyro
        rmse = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

        i_norm = np.linalg.norm(i_gyro, axis=1)
        g_norm = np.linalg.norm(g_gyro, axis=1)
        corr = np.corrcoef(i_norm, g_norm)[0, 1] if np.std(i_norm) > 1e-6 else 0

        # 重力误差（应用旋转）
        acc_w = g_rot.apply(i_acce)
        mean_acc = np.mean(acc_w, axis=0)
        g_mag = np.linalg.norm(mean_acc)
        angle_err = np.degrees(np.arccos(
            np.clip(abs(np.dot(mean_acc / g_mag, self.z_axis)), -1, 1)
        ))

        # 时移敏感指标
        i_diff = np.diff(i_norm)
        g_diff = np.diff(g_norm)
        sign_consistency = np.mean(i_diff * g_diff > 0) if len(i_diff) > 0 else 0

        i_peaks, _ = find_peaks(i_norm, height=np.mean(i_norm) + np.std(i_norm))
        g_peaks, _ = find_peaks(g_norm, height=np.mean(g_norm) + np.std(g_norm))
        peak_alignment = 0.0
        if len(i_peaks) > 0 and len(g_peaks) > 0:
            peak_distances = []
            for ip in i_peaks:
                if len(g_peaks) > 0:
                    closest_gp = g_peaks[np.argmin(np.abs(g_peaks - ip))]
                    peak_distances.append(abs(ip - closest_gp))
            peak_alignment = np.mean(peak_distances) if peak_distances else 0

        # 频域相关性
        min_len = min(len(i_norm), len(g_norm))
        i_fft = np.abs(fft(i_norm[:min_len]))
        g_fft = np.abs(fft(g_norm[:min_len]))
        energy_corr = np.corrcoef(i_fft, g_fft)[0, 1] if np.std(i_fft) > 1e-6 else 0

        return {
            "RMSE": rmse,
            "Corr": corr,
            "GravErr": angle_err,
            "GravMag": g_mag,
            "SignConsistency": sign_consistency,
            "PeakAlignment": peak_alignment,
            "EnergyCorr": energy_corr
        }

    def compute_metrics_original(self, i_gyro, i_acce, g_gyro) -> Dict[str, float]:
        """计算评估指标（不应用g_rot旋转 - Original阶段）

        假设重力方向在IMU坐标系的Z轴正向
        """
        diff = i_gyro - g_gyro
        rmse = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

        i_norm = np.linalg.norm(i_gyro, axis=1)
        g_norm = np.linalg.norm(g_gyro, axis=1)
        corr = np.corrcoef(i_norm, g_norm)[0, 1] if np.std(i_norm) > 1e-6 else 0

        # 重力误差（不应用旋转，假设重力在IMU坐标系Z轴负向）
        mean_acc = np.mean(i_acce, axis=0)
        g_mag = np.linalg.norm(mean_acc)
        # 使用配置的Z轴方向（向上或向下）
        angle_err = np.degrees(np.arccos(
            np.clip(abs(np.dot(mean_acc / g_mag, self.z_axis)), -1, 1)
        ))

        # 时移敏感指标
        i_diff = np.diff(i_norm)
        g_diff = np.diff(g_norm)
        sign_consistency = np.mean(i_diff * g_diff > 0) if len(i_diff) > 0 else 0

        i_peaks, _ = find_peaks(i_norm, height=np.mean(i_norm) + np.std(i_norm))
        g_peaks, _ = find_peaks(g_norm, height=np.mean(g_norm) + np.std(g_norm))
        peak_alignment = 0.0
        if len(i_peaks) > 0 and len(g_peaks) > 0:
            peak_distances = []
            for ip in i_peaks:
                if len(g_peaks) > 0:
                    closest_gp = g_peaks[np.argmin(np.abs(g_peaks - ip))]
                    peak_distances.append(abs(ip - closest_gp))
            peak_alignment = np.mean(peak_distances) if peak_distances else 0

        # 频域相关性
        min_len = min(len(i_norm), len(g_norm))
        i_fft = np.abs(fft(i_norm[:min_len]))
        g_fft = np.abs(fft(g_norm[:min_len]))
        energy_corr = np.corrcoef(i_fft, g_fft)[0, 1] if np.std(i_fft) > 1e-6 else 0

        return {
            "RMSE": rmse,
            "Corr": corr,
            "GravErr": angle_err,
            "GravMag": g_mag,
            "SignConsistency": sign_consistency,
            "PeakAlignment": peak_alignment,
            "EnergyCorr": energy_corr
        }

    def time_align_data(self, imu_data, imu_t, gt_t, method="interp+shift"):
        """时间对齐"""
        interp_fun = interp1d(
            imu_t, imu_data, axis=0, kind='linear',
            bounds_error=False, fill_value='extrapolate'
        )
        imu_interp = interp_fun(gt_t)

        if method != "interp+shift":
            return imu_interp, 0.0, {}

        # 互相关分析
        imu_norm = np.linalg.norm(imu_interp, axis=1)
        gt_gyro_for_corr = self.g_gyro_eval[:len(imu_norm)]
        gt_norm = np.linalg.norm(gt_gyro_for_corr, axis=1)

        corr = correlate(
            imu_norm - np.mean(imu_norm),
            gt_norm - np.mean(gt_norm),
            mode='full'
        )
        lags = np.arange(-len(imu_norm) + 1, len(gt_norm))
        best_lag = lags[np.argmax(corr)]

        dt_gt = np.mean(np.diff(gt_t))
        time_shift_us = best_lag * dt_gt

        # 应用时移
        if abs(best_lag) < len(imu_interp):
            if best_lag > 0:
                imu_aligned = np.roll(imu_interp, -best_lag, axis=0)
                imu_aligned[-best_lag:] = imu_aligned[-best_lag - 1:-1]
            else:
                imu_aligned = np.roll(imu_interp, -best_lag, axis=0)
                imu_aligned[:-best_lag] = imu_aligned[1:-best_lag + 1]
        else:
            imu_aligned = imu_interp
            time_shift_us = 0.0

        debug_info = {
            'time_shift_ms': float(time_shift_us / 1000),
            'best_lag': int(best_lag)
        }

        return imu_aligned, time_shift_us, debug_info

    def evaluate(self) -> Tuple[EvaluationMetrics, Dict[str, Any]]:
        """执行评估

        Returns:
            (metrics, debug_data): metrics 为评估指标，debug_data 包含中间数据用于可视化
        """
        # 0. Original阶段（完全不应用g_rot旋转）
        m_original = self.compute_metrics_original(
            self.i_gyro_eval, self.i_acce_eval,
            self.g_gyro_eval
        )

        # 1. Raw阶段（应用g_rot旋转）
        m_raw = self.compute_metrics(
            self.i_gyro_eval, self.i_acce_eval,
            self.g_gyro_eval, self.g_rots_eval
        )

        # 2. 时间对齐
        i_gyro_interp, time_shift_us, debug_info = self.time_align_data(
            self.i_gyro_eval, self.i_t_eval, self.g_t_eval, method="interp+shift"
        )

        i_t_shifted = self.i_t_eval - time_shift_us
        i_gyro_synced, _, _ = self.time_align_data(
            self.i_gyro_eval, i_t_shifted, self.g_t_eval, method="interp"
        )
        i_acce_synced, _, _ = self.time_align_data(
            self.i_acce_eval, i_t_shifted, self.g_t_eval, method="interp"
        )

        m_time = self.compute_metrics(
            i_gyro_synced, i_acce_synced,
            self.g_gyro_eval, self.g_rots_eval
        )

        # 3. 空间校准（可选）
        if self.enable_calibration:
            weights = np.linalg.norm(self.g_gyro_eval, axis=1)
            mask = weights > 0.5
            if np.sum(mask) < 100:
                mask = weights > 0

            try:
                R_calib, _ = Rotation.align_vectors(
                    self.g_gyro_eval[mask], i_gyro_synced[mask]
                )
                euler = R_calib.as_euler('xyz', degrees=True)
            except Exception:
                R_calib = Rotation.identity()
                euler = np.zeros(3)

            # 4. 最终指标（应用空间校准）
            i_gyro_final = R_calib.apply(i_gyro_synced)
            i_acce_final = R_calib.apply(i_acce_synced)

            m_final = self.compute_metrics(
                i_gyro_final, i_acce_final,
                self.g_gyro_eval, self.g_rots_eval
            )
        else:
            # 不启用空间校准，使用 time 对齐后的结果作为 final
            R_calib = Rotation.identity()
            euler = np.zeros(3)
            m_final = m_time

        # 计算加速度旋转（用于可视化）
        # 使用第一个GT姿态作为参考（或者使用平均姿态）
        g_rot_ref = self.g_rots_eval[0]  # 使用第一个姿态作为参考
        acc_w = g_rot_ref.apply(i_acce_synced)

        # 5. 构造结果
        metrics = EvaluationMetrics(
            unit_name=self.name,
            dataset=self.dataset,
            group=self.group,
            # Original阶段
            original_rmse=m_original['RMSE'],
            original_corr=m_original['Corr'],
            original_grav_err=m_original['GravErr'],
            original_grav_mag=m_original['GravMag'],
            # Raw阶段
            raw_rmse=m_raw['RMSE'],
            raw_corr=m_raw['Corr'],
            raw_grav_err=m_raw['GravErr'],
            raw_grav_mag=m_raw['GravMag'],
            # Time阶段
            time_rmse=m_time['RMSE'],
            time_corr=m_time['Corr'],
            time_grav_err=m_time['GravErr'],
            time_grav_mag=m_time['GravMag'],
            time_shift_ms=debug_info['time_shift_ms'],
            # Final阶段
            final_rmse=m_final['RMSE'],
            final_corr=m_final['Corr'],
            final_grav_err=m_final['GravErr'],
            final_grav_mag=m_final['GravMag'],
            calib_euler_x=euler[0],
            calib_euler_y=euler[1],
            calib_euler_z=euler[2],
            # 时移敏感指标
            sign_consistency=m_final['SignConsistency'],
            peak_alignment=m_final['PeakAlignment'],
            energy_corr=m_final['EnergyCorr']
        )

        # 调试数据（用于可视化）
        debug_data = {
            'i_gyro_raw': self.i_gyro_eval[:len(i_gyro_interp)],
            'i_gyro_aligned': i_gyro_interp,
            'g_gyro': self.g_gyro_eval[:len(i_gyro_interp)],
            'time_shift_ms': debug_info['time_shift_ms'],
            't_us': self.g_t_eval[:len(i_gyro_interp)],  # GT 时间轴
            'i_acce_synced': i_acce_synced[:len(i_gyro_interp)],  # IMU坐标系加速度
            'acc_w': acc_w[:len(i_gyro_interp)]  # 世界坐标系加速度
        }

        return metrics, debug_data

    def run_and_calibrate(self, output_root: Path = None):
        """执行评估并保存校准后的数据（兼容旧 API）

        Args:
            output_root: 输出目录路径

        Returns:
            metrics: 评估指标
        """
        from base.serialize import ImuDataSerializer, PosesDataSerializer
        from base.datatype import ImuData
        from datetime import datetime

        # 执行评估
        metrics, _ = self.evaluate()

        # 打印结果
        print(f"\n{'='*80}")
        print(f"📊 EVALUATION RESULT: {self.name}")
        print(f"{'='*80}")
        print(f"  Original RMSE:    {metrics.original_rmse:.4f} rad/s")
        print(f"  Raw (GT) RMSE:    {metrics.raw_rmse:.4f} rad/s")
        print(f"  Time RMSE:        {metrics.time_rmse:.4f} rad/s (shift: {metrics.time_shift_ms:.2f} ms)")
        print(f"  Final (opt) RMSE: {metrics.final_rmse:.4f} rad/s")
        print(f"  Correlation:      {metrics.final_corr:.4f}")
        print(f"  Gravity Err:      {metrics.final_grav_err:.2f}°")
        print(f"  Calibration:      X={metrics.calib_euler_x:.2f}° Y={metrics.calib_euler_y:.2f}° Z={metrics.calib_euler_z:.2f}°")
        print(f"{'='*80}")

        # 保存校准数据
        if output_root:
            save_dir = output_root / self.name
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"-> Saving calibrated data to: {save_dir}")

            # 1. 计算校准参数
            time_shift_us = metrics.time_shift_ms * 1000
            euler = np.array([metrics.calib_euler_x, metrics.calib_euler_y, metrics.calib_euler_z])
            R_calib = Rotation.from_euler('xyz', euler, degrees=True)

            # 2. 对全量 IMU 数据做时间对齐（使用 interp 方法，不带 shift）
            i_t_shifted = self.imu.t_us - time_shift_us

            # 重新插值到 GT 时间轴
            interp_fun_gyro = interp1d(
                i_t_shifted, self.imu.gyro, axis=0, kind='linear',
                bounds_error=False, fill_value='extrapolate'
            )
            interp_fun_acce = interp1d(
                i_t_shifted, self.imu.acce, axis=0, kind='linear',
                bounds_error=False, fill_value='extrapolate'
            )

            i_gyro_time_aligned = interp_fun_gyro(self.gt.t_us)
            i_acce_time_aligned = interp_fun_acce(self.gt.t_us)

            # 3. 应用坐标系校准
            new_gyro = R_calib.apply(i_gyro_time_aligned)
            new_acce = R_calib.apply(i_acce_time_aligned)
            new_ahrs = R_calib * self.imu.ahrs

            # 4. 构造新的 ImuData（使用 GT 的时间轴）
            # magnetometer 也需要插值
            interp_fun_magn = interp1d(
                i_t_shifted, self.imu.magn, axis=0, kind='linear',
                bounds_error=False, fill_value='extrapolate'
            )
            new_magn = interp_fun_magn(self.gt.t_us)

            imu_new = ImuData(
                self.gt.t_us,
                new_gyro, new_acce, new_ahrs, new_magn
            )

            # 5. 保存 IMU 和 GT 数据
            ImuDataSerializer(imu_new).save(save_dir / "imu.csv")
            PosesDataSerializer(self.gt).save(save_dir / "gt.csv")

            # 6. 保存校准参数
            calib_info = {
                "time_shift_us": float(time_shift_us),
                "rotation_euler_xyz_deg": euler.tolist(),
                "rotation_quat_xyzw": R_calib.as_quat().tolist()
            }
            np.save(save_dir / "calibration_params.npy", calib_info)

            # 7. 保存评估结果（TXT + JSON）
            self._save_evaluation_report(save_dir, metrics, calib_info)

            print("   ✅ Save complete (time alignment + spatial calibration)")

        return metrics

    def _save_evaluation_report(self, save_dir: Path, metrics: EvaluationMetrics, calib_info: dict):
        """保存评估报告（TXT + JSON）"""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # === 1. 保存 TXT 报告（人类可读） ===
        txt_path = save_dir / "evaluation.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"EVALUATION RESULT: {metrics.unit_name}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {timestamp}\n\n")

            # RMSE
            f.write("--- RMSE (rad/s) ---\n")
            f.write(f"  Original:    {metrics.original_rmse:.4f}\n")
            f.write(f"  Raw (GT):    {metrics.raw_rmse:.4f}\n")
            f.write(f"  Time:        {metrics.time_rmse:.4f}  (shift: {metrics.time_shift_ms:.2f} ms)\n")
            f.write(f"  Final (opt): {metrics.final_rmse:.4f}")
            if metrics.original_rmse > 0:
                improvement = (metrics.original_rmse - metrics.final_rmse) / metrics.original_rmse * 100
                f.write(f"  ↓ {improvement:.1f}% improvement from Original")
            f.write("\n\n")

            # Correlation
            f.write("--- Correlation ---\n")
            f.write(f"  Original:    {metrics.original_corr:.4f}\n")
            f.write(f"  Raw (GT):    {metrics.raw_corr:.4f}\n")
            f.write(f"  Time:        {metrics.time_corr:.4f}\n")
            f.write(f"  Final (opt): {metrics.final_corr:.4f}\n\n")

            # Gravity Error
            f.write("--- Gravity Error (deg) ---\n")
            f.write(f"  Original:    {metrics.original_grav_err:.2f}°\n")
            f.write(f"  Raw (GT):    {metrics.raw_grav_err:.2f}°\n")
            f.write(f"  Time:        {metrics.time_grav_err:.2f}°\n")
            f.write(f"  Final (opt): {metrics.final_grav_err:.2f}°\n\n")

            # Gravity Magnitude
            f.write("--- Gravity Magnitude (m/s²) ---\n")
            f.write(f"  Original:    {metrics.original_grav_mag:.2f}\n")
            f.write(f"  Raw (GT):    {metrics.raw_grav_mag:.2f}\n")
            f.write(f"  Time:        {metrics.time_grav_mag:.2f}\n")
            f.write(f"  Final (opt): {metrics.final_grav_mag:.2f}\n\n")

            # Calibration
            f.write("--- Spatial Calibration ---\n")
            f.write(f"  Euler (XYZ):  {metrics.calib_euler_x:.2f}°, {metrics.calib_euler_y:.2f}°, {metrics.calib_euler_z:.2f}°\n")
            f.write(f"  Time Shift:  {metrics.time_shift_ms:.2f} ms\n\n")

            # Time-Sensitive Metrics
            f.write("--- Time-Sensitive Metrics ---\n")
            f.write(f"  Sign Consistency:  {metrics.sign_consistency:.1%}\n")
            f.write(f"  Peak Alignment:    {metrics.peak_alignment:.1f} samples\n")
            f.write(f"  Energy Correlation: {metrics.energy_corr:.4f}\n")

            f.write("=" * 80 + "\n")

        print(f"   📄 Evaluation report saved: {txt_path}")


# ==================== 可视化器 ====================

class Visualizer:
    """可视化器 - 生成图表和 rerun 视图"""

    def __init__(self, save_dir: Path, save_to_unit_dir: bool = False):
        """
        Args:
            save_dir: 输出目录
            save_to_unit_dir: 是否将图表保存到各自序列目录下
        """
        self.save_dir = Path(save_dir)
        self.save_to_unit_dir = save_to_unit_dir
        if not save_to_unit_dir:
            # 统一保存模式：所有图表保存到一个目录
            self.plot_dir = self.save_dir / "plots"
            self.plot_dir.mkdir(parents=True, exist_ok=True)

    def plot_time_alignment(
        self,
        name: str,
        i_gyro_raw: np.ndarray,
        i_gyro_aligned: np.ndarray,
        g_gyro: np.ndarray,
        time_shift_ms: float,
        t_us: np.ndarray = None,
        unit_path: Path = None
    ):
        """绘制时间对齐对比图

        Args:
            t_us: 时间轴数据（微秒），如果为 None 则使用索引生成
            unit_path: 序列路径（如果 save_to_unit_dir=True，需要传入）
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        if t_us is not None:
            t_sec = t_us[:len(i_gyro_raw)] / 1e6
            t_sec = t_sec - t_sec[0]  # 从 0 开始
        else:
            t_sec = np.arange(len(i_gyro_raw)) * 0.01  # 默认 10ms 采样

        imu_norm_raw = np.linalg.norm(i_gyro_raw, axis=1)
        imu_norm_aligned = np.linalg.norm(i_gyro_aligned, axis=1)
        gt_norm = np.linalg.norm(g_gyro[:len(i_gyro_raw)], axis=1)

        # 子图1：对齐前
        axes[0].plot(t_sec, imu_norm_raw, label='IMU (Before)', alpha=0.7, linewidth=1)
        axes[0].plot(t_sec, gt_norm, label='Ground Truth', alpha=0.7, linewidth=1)
        axes[0].set_ylabel('Angular Velocity (rad/s)')
        axes[0].set_title(f'Time Alignment: BEFORE - {name}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 子图2：对齐后
        axes[1].plot(t_sec, imu_norm_aligned, label='IMU (After)', alpha=0.7, linewidth=1, color='orange')
        axes[1].plot(t_sec, gt_norm, label='Ground Truth', alpha=0.7, linewidth=1)
        axes[1].set_ylabel('Angular Velocity (rad/s)')
        axes[1].set_title(f'Time Alignment: AFTER (Shift={time_shift_ms:.2f}ms) - {name}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 子图3：差异
        diff_before = imu_norm_raw - gt_norm
        diff_after = imu_norm_aligned - gt_norm
        axes[2].plot(t_sec, diff_before, label='Before', alpha=0.7, linewidth=1)
        axes[2].plot(t_sec, diff_after, label='After', alpha=0.7, linewidth=1, color='orange')
        axes[2].axhline(0, color='black', linestyle='--', linewidth=0.8)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Difference (rad/s)')
        axes[2].set_title('Alignment Effect')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        # 确定保存路径
        if self.save_to_unit_dir and unit_path:
            save_dir = unit_path
        else:
            save_dir = self.plot_dir
            save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / "time_alignment.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  📈 Time alignment plot saved: {save_path}")

    def plot_acceleration_rotation(
        self,
        name: str,
        i_acce: np.ndarray,
        acc_w: np.ndarray,
        t_us: np.ndarray = None,
        unit_path: Path = None
    ):
        """绘制加速度旋转对比图（IMU坐标系 vs 世界坐标系）

        Args:
            name: 序列名称
            i_acce: IMU坐标系下的加速度 (N, 3)
            acc_w: 世界坐标系下的加速度 (N, 3)
            t_us: 时间轴数据（微秒），如果为 None 则使用索引生成
            unit_path: 序列路径（如果 save_to_unit_dir=True，需要传入）
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        if t_us is not None:
            t_sec = t_us[:len(i_acce)] / 1e6
            t_sec = t_sec - t_sec[0]  # 从 0 开始
        else:
            t_sec = np.arange(len(i_acce)) * 0.01  # 默认 10ms 采样

        colors = ['red', 'green', 'blue']
        axis_names = ['X', 'Y', 'Z']

        # 子图1: IMU坐标系下的加速度
        for i in range(3):
            axes[0].plot(t_sec, i_acce[:, i], label=f'Acc_{axis_names[i]} (IMU)',
                        color=colors[i], alpha=0.7, linewidth=1)
        axes[0].set_ylabel('Acceleration (m/s²)')
        axes[0].set_title(f'Acceleration in IMU Frame - {name}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 子图2: 世界坐标系下的加速度
        for i in range(3):
            axes[1].plot(t_sec, acc_w[:, i], label=f'Acc_{axis_names[i]} (World)',
                        color=colors[i], alpha=0.7, linewidth=1)
        # 标记重力方向（世界坐标系Z轴）
        axes[1].axhline(9.81, color='purple', linestyle='--', linewidth=1.5,
                       alpha=0.5, label='Gravity (9.81 m/s²)')
        axes[1].set_ylabel('Acceleration (m/s²)')
        axes[1].set_title(f'Acceleration in World Frame - {name}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 子图3: 对比（各轴模长）
        i_acce_norm = np.linalg.norm(i_acce, axis=1)
        acc_w_norm = np.linalg.norm(acc_w, axis=1)
        axes[2].plot(t_sec, i_acce_norm, label='IMU Frame Norm', alpha=0.7, linewidth=1.5, color='orange')
        axes[2].plot(t_sec, acc_w_norm, label='World Frame Norm', alpha=0.7, linewidth=1.5, color='cyan')
        axes[2].axhline(9.81, color='purple', linestyle='--', linewidth=1.5,
                       alpha=0.5, label='Expected Gravity')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Acceleration Norm (m/s²)')
        axes[2].set_title('Acceleration Magnitude Comparison')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        # 确定保存路径
        if self.save_to_unit_dir and unit_path:
            save_dir = unit_path
        else:
            save_dir = self.plot_dir
            save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / "acceleration_rotation.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  📈 Acceleration rotation plot saved: {save_path}")

    def plot_trajectory_2d(
        self,
        name: str,
        gt_poses,
        imu_poses=None,
        unit_path: Path = None
    ):
        """绘制 2D 轨迹图（俯视）

        Args:
            unit_path: 序列路径（如果 save_to_unit_dir=True，需要传入）
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        gt_ps = gt_poses.ps
        ax.plot(gt_ps[:, 0], gt_ps[:, 1], 'g-', label='Ground Truth', linewidth=2)
        ax.plot(gt_ps[0, 0], gt_ps[0, 1], 'go', markersize=10, label='Start')
        ax.plot(gt_ps[-1, 0], gt_ps[-1, 1], 'rx', markersize=12, markeredgewidth=3, label='End')

        if imu_poses is not None:
            imu_ps = imu_poses.ps
            ax.plot(imu_ps[:, 0], imu_ps[:, 1], 'b--', label='IMU', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'2D Trajectory - {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # 确定保存路径
        if self.save_to_unit_dir and unit_path:
            save_dir = unit_path
        else:
            save_dir = self.plot_dir
            save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / "trajectory_2d.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  📈 2D trajectory plot saved: {save_path}")

    def launch_rerun(
        self,
        name: str,
        unit: UnitData,
        imu_calib=None
    ):
        """启动 rerun 3D 可视化"""
        try:
            RerunView().add_spatial_view().add_imu_view(tags=[name]).send(name)

            # 记录 GT 轨迹
            send_pose_data(unit.gt_data, tag="Groundtruth", color=[0, 255, 0])

            # 记录 IMU 数据
            send_imu_data(unit.imu_data, tag=name)

            print(f"  🎥 Rerun viewer launched: {name}")
        except Exception as e:
            print(f"  ⚠️ Rerun visualization failed: {e}")


# ==================== 指标收集器 ====================

class MetricsCollector:
    """指标收集器 - 保存评估结果到序列目录"""

    def __init__(self):
        self.metrics_list: List[EvaluationMetrics] = []
        self.rotation_eulers: List[np.ndarray] = []  # 存储所有校准的欧拉角
        self.unit_names: List[str] = []  # 存储对应的序列名称

    def add(self, metrics: EvaluationMetrics, unit_path: Path, unit: UnitData = None):
        """添加指标并保存到序列目录

        Args:
            metrics: 评估指标
            unit_path: 序列路径
            unit: 可选，单元数据（用于保存统计特性）
        """
        self.metrics_list.append(metrics)

        # 收集旋转欧拉角用于一致性分析
        euler = np.array([
            metrics.calib_euler_x,
            metrics.calib_euler_y,
            metrics.calib_euler_z
        ])
        self.rotation_eulers.append(euler)
        self.unit_names.append(metrics.unit_name)

        self._save_unit_report(unit_path, metrics)
        if unit:
            self._save_unit_statistics(unit_path, metrics, unit)

    def _save_unit_report(self, unit_path: Path, metrics: EvaluationMetrics):
        """在序列目录下保存评估报告"""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 保存 TXT
        txt_path = unit_path / "evaluation.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"EVALUATION RESULT: {metrics.unit_name}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {timestamp}\n\n")

            # RMSE
            f.write("--- RMSE (rad/s) ---\n")
            f.write(f"  Original:    {metrics.original_rmse:.4f}\n")
            f.write(f"  Raw (GT):    {metrics.raw_rmse:.4f}\n")
            f.write(f"  Time:        {metrics.time_rmse:.4f}  (shift: {metrics.time_shift_ms:.2f} ms)\n")
            f.write(f"  Final (opt): {metrics.final_rmse:.4f}")
            if metrics.original_rmse > 0:
                improvement = (metrics.original_rmse - metrics.final_rmse) / metrics.original_rmse * 100
                f.write(f"  ↓ {improvement:.1f}% improvement from Original")
            f.write("\n\n")

            # Correlation
            f.write("--- Correlation ---\n")
            f.write(f"  Original:    {metrics.original_corr:.4f}\n")
            f.write(f"  Raw (GT):    {metrics.raw_corr:.4f}\n")
            f.write(f"  Time:        {metrics.time_corr:.4f}\n")
            f.write(f"  Final (opt): {metrics.final_corr:.4f}\n\n")

            # Gravity Error
            f.write("--- Gravity Error (deg) ---\n")
            f.write(f"  Original:    {metrics.original_grav_err:.2f}°\n")
            f.write(f"  Raw (GT):    {metrics.raw_grav_err:.2f}°\n")
            f.write(f"  Time:        {metrics.time_grav_err:.2f}°\n")
            f.write(f"  Final (opt): {metrics.final_grav_err:.2f}°\n\n")

            # Gravity Magnitude
            f.write("--- Gravity Magnitude (m/s²) ---\n")
            f.write(f"  Original:    {metrics.original_grav_mag:.2f}\n")
            f.write(f"  Raw (GT):    {metrics.raw_grav_mag:.2f}\n")
            f.write(f"  Time:        {metrics.time_grav_mag:.2f}\n")
            f.write(f"  Final (opt): {metrics.final_grav_mag:.2f}\n\n")

            # Calibration
            f.write("--- Spatial Calibration ---\n")
            f.write(f"  Euler (XYZ):  {metrics.calib_euler_x:.2f}°, {metrics.calib_euler_y:.2f}°, {metrics.calib_euler_z:.2f}°\n")
            f.write(f"  Time Shift:  {metrics.time_shift_ms:.2f} ms\n\n")

            # Time-Sensitive Metrics
            f.write("--- Time-Sensitive Metrics ---\n")
            f.write(f"  Sign Consistency:  {metrics.sign_consistency:.1%}\n")
            f.write(f"  Peak Alignment:    {metrics.peak_alignment:.1f} samples\n")
            f.write(f"  Energy Correlation: {metrics.energy_corr:.4f}\n")

            f.write("=" * 80 + "\n")

        print(f"  📄 Evaluation report saved: {txt_path}")

    def _save_unit_statistics(self, unit_path: Path, metrics: EvaluationMetrics, unit: UnitData):
        """保存序列的统计特性到JSON

        Args:
            unit_path: 序列路径
            metrics: 评估指标
            unit: 单元数据
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 计算统计特性
        def compute_array_stats(arr: np.ndarray, axis_names: List[str]) -> Dict:
            """计算数组统计特性"""
            if arr is None or len(arr) == 0:
                return {}

            stats = {}
            for i, name in enumerate(axis_names):
                data = arr[:, i] if arr.ndim > 1 else arr
                stats[name] = {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "median": float(np.median(data)),
                    "range": float(np.max(data) - np.min(data))
                }
            return stats

        # IMU 统计
        imu_stats = {
            "gyro": compute_array_stats(unit.imu_data.gyro, ['x', 'y', 'z']),
            "acce": compute_array_stats(unit.imu_data.acce, ['x', 'y', 'z']),
            "magn": compute_array_stats(unit.imu_data.magn, ['x', 'y', 'z']) if unit.imu_data.magn is not None else {}
        }

        # GT 统计
        gt_stats = {}
        if unit.gt_data is not None:
            # 位置统计
            if hasattr(unit.gt_data, 'ps') and unit.gt_data.ps is not None:
                gt_stats["position"] = compute_array_stats(unit.gt_data.ps, ['x', 'y', 'z'])

            # 速度统计（如果存在）
            if hasattr(unit.gt_data, 'vs') and unit.gt_data.vs is not None:
                gt_stats["velocity"] = compute_array_stats(unit.gt_data.vs, ['x', 'y', 'z'])

        # 时间统计
        t_sec = unit.imu_data.t_us / 1e6
        time_stats = {
            "duration_sec": float(t_sec[-1] - t_sec[0]),
            "start_time_sec": float(t_sec[0]),
            "end_time_sec": float(t_sec[-1]),
            "num_samples": int(len(unit.imu_data.t_us)),
            "sample_rate_hz": float(len(unit.imu_data.t_us) / (t_sec[-1] - t_sec[0])) if t_sec[-1] > t_sec[0] else 0.0
        }

        # 组装完整的统计信息
        statistics = {
            "unit_name": metrics.unit_name,
            "timestamp": timestamp,
            "time_info": time_stats,
            "imu_statistics": imu_stats,
            "gt_statistics": gt_stats,
            "evaluation_metrics": {
                "rmse": {
                    "original": metrics.original_rmse,
                    "raw": metrics.raw_rmse,
                    "time_aligned": metrics.time_rmse,
                    "final": metrics.final_rmse
                },
                "correlation": {
                    "original": metrics.original_corr,
                    "raw": metrics.raw_corr,
                    "time_aligned": metrics.time_corr,
                    "final": metrics.final_corr
                },
                "gravity_error_deg": {
                    "original": metrics.original_grav_err,
                    "raw": metrics.raw_grav_err,
                    "time_aligned": metrics.time_grav_err,
                    "final": metrics.final_grav_err
                },
                "gravity_magnitude": {
                    "original": metrics.original_grav_mag,
                    "raw": metrics.raw_grav_mag,
                    "time_aligned": metrics.time_grav_mag,
                    "final": metrics.final_grav_mag
                },
                "spatial_calibration": {
                    "euler_xyz_deg": [metrics.calib_euler_x, metrics.calib_euler_y, metrics.calib_euler_z],
                    "time_shift_ms": metrics.time_shift_ms
                },
                "time_sensitive_metrics": {
                    "sign_consistency": metrics.sign_consistency,
                    "peak_alignment_samples": metrics.peak_alignment,
                    "energy_correlation": metrics.energy_corr
                }
            }
        }

        # 保存JSON
        json_path = unit_path / "statistics.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)

        print(f"  📊 Statistics saved: {json_path}")

    def print_summary(self):
        """打印评估摘要"""
        if not self.metrics_list:
            return

        print("\n📊 Evaluation Summary:")
        print("=" * 100)
        print(f"{'Unit':<30} | {'Orig RMSE':<12} | {'Raw(GT) RMSE':<14} | {'Final(opt) RMSE':<16} | {'Shift':<10}")
        print("-" * 100)
        for m in self.metrics_list:
            print(f"{m.unit_name:<30} | {m.original_rmse:<12.4f} | {m.raw_rmse:<14.4f} | {m.final_rmse:<16.4f} | {m.time_shift_ms:<10.2f}")
        print("=" * 100)

    def check_rotation_consistency(self, save_dir: Path = None):
        """检查多个序列间的旋转一致性

        Args:
            save_dir: 可选，保存可视化图表的目录
        """
        if len(self.rotation_eulers) < 2:
            print("\n⚠️  需要至少2个序列才能进行旋转一致性分析")
            return

        eulers = np.array(self.rotation_eulers)  # shape: (N, 3)

        # 计算统计信息
        mean_euler = np.mean(eulers, axis=0)
        std_euler = np.std(eulers, axis=0)
        range_euler = np.max(eulers, axis=0) - np.min(eulers, axis=0)

        # 计算旋转矩阵之间的角度差异
        rotations = [
            Rotation.from_euler('xyz', euler, degrees=True)
            for euler in eulers
        ]

        # 计算相对于平均旋转的角度偏差
        if len(eulers) >= 2:
            # 使用第一个作为参考，或者计算平均旋转
            mean_rot = Rotation.from_euler('xyz', mean_euler, degrees=True)
            angle_deviations = []
            for rot in rotations:
                # 计算相对旋转的角度
                rel_rot = mean_rot.inv() * rot
                angle = np.degrees(rel_rot.magnitude())
                angle_deviations.append(angle)

            mean_angle_dev = np.mean(angle_deviations)
            max_angle_dev = np.max(angle_deviations)

        print("\n" + "=" * 80)
        print("🔄 IMU-GT 旋转一致性分析")
        print("=" * 80)

        print("\n📊 欧拉角统计 (XYZ, degrees):")
        print("-" * 80)
        print(f"{'轴':<10} | {'均值':<12} | {'标准差':<12} | {'范围':<12} | {'一致性判定'}")
        print("-" * 80)

        axes = ['X', 'Y', 'Z']
        consistency_judgment = []
        for i, axis in enumerate(axes):
            std = std_euler[i]
            range_val = range_euler[i]

            if std < 5:
                judgment = "✅ 优秀 (<5°)"
            elif std < 10:
                judgment = "⚠️  中等 (5-10°)"
            else:
                judgment = "❌ 差 (>10°)"

            consistency_judgment.append(std < 10)  # 10度以下认为可接受

            print(f"{axis:<10} | {mean_euler[i]:>10.2f}° | {std:>10.2f}° | {range_val:>10.2f}° | {judgment}")

        print("-" * 80)

        if len(eulers) >= 2:
            print(f"\n📐 相对于平均旋转的角度偏差:")
            print(f"  平均偏差: {mean_angle_dev:.2f}°")
            print(f"  最大偏差: {max_angle_dev:.2f}°")

            if mean_angle_dev < 5:
                print(f"  判定: ✅ 刚性连接 (偏差 < 5°)")
            elif mean_angle_dev < 15:
                print(f"  判定: ⚠️  可能存在柔性连接或安装偏差 (偏差 5-15°)")
            else:
                print(f"  判定: ❌ 非刚性连接，每个序列需单独标定 (偏差 > 15°)")

        # 详细列表
        print(f"\n📋 各序列校准角度详情:")
        print("-" * 80)
        for i, (name, euler) in enumerate(zip(self.unit_names, eulers)):
            print(f"  {i+1}. {name:<30} Euler({euler[0]:>7.2f}°, {euler[1]:>7.2f}°, {euler[2]:>7.2f}°)")
        print("-" * 80)

        # 可视化
        if save_dir:
            self._plot_rotation_consistency(eulers, self.unit_names, save_dir)

        print("=" * 80)

        return {
            'mean_euler': mean_euler,
            'std_euler': std_euler,
            'range_euler': range_euler,
            'mean_angle_deviation': mean_angle_dev if len(eulers) >= 2 else None,
            'is_rigid': mean_angle_dev < 5 if len(eulers) >= 2 else None
        }

    def _plot_rotation_consistency(self, eulers: np.ndarray, names: List[str], save_dir: Path):
        """绘制旋转一致性可视化图表

        Args:
            eulers: 欧拉角数组 (N, 3)
            names: 序列名称列表
            save_dir: 保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = plt.cm.tab10(np.linspace(0, 1, len(eulers)))

        # 子图1: 欧拉角散点图 (X vs Y)
        ax = axes[0, 0]
        for i, (euler, name) in enumerate(zip(eulers, names)):
            ax.scatter(euler[0], euler[1], c=[colors[i]], s=100, alpha=0.7,
                      label=f'{name[:15]}...' if len(name) > 15 else name)
        ax.set_xlabel('Euler X (deg)')
        ax.set_ylabel('Euler Y (deg)')
        ax.set_title('Euler Angles: X vs Y')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

        # 子图2: 欧拉角散点图 (Y vs Z)
        ax = axes[0, 1]
        for i, (euler, name) in enumerate(zip(eulers, names)):
            ax.scatter(euler[1], euler[2], c=[colors[i]], s=100, alpha=0.7,
                      label=f'{name[:15]}...' if len(name) > 15 else name)
        ax.set_xlabel('Euler Y (deg)')
        ax.set_ylabel('Euler Z (deg)')
        ax.set_title('Euler Angles: Y vs Z')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

        # 子图3: 各轴的箱线图
        ax = axes[1, 0]
        data_to_plot = [eulers[:, 0], eulers[:, 1], eulers[:, 2]]
        bp = ax.boxplot(data_to_plot, labels=['X', 'Y', 'Z'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['red', 'green', 'blue']):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
        ax.set_ylabel('Euler Angle (deg)')
        ax.set_title('Euler Angles Distribution (Box Plot)')
        ax.grid(True, alpha=0.3, axis='y')

        # 子图4: 标准差条形图
        ax = axes[1, 1]
        stds = np.std(eulers, axis=0)
        bars = ax.bar(['X', 'Y', 'Z'], stds, color=['red', 'green', 'blue'], alpha=0.7)
        ax.axhline(y=5, color='orange', linestyle='--', linewidth=1.5, label='Good threshold (5°)')
        ax.axhline(y=10, color='red', linestyle='--', linewidth=1.5, label='Acceptable threshold (10°)')
        ax.set_ylabel('Standard Deviation (deg)')
        ax.set_title('Euler Angles Standard Deviation')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 在条形上添加数值
        for bar, std in zip(bars, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{std:.2f}°',
                   ha='center', va='bottom')

        plt.tight_layout()

        # 保存图片
        save_path = save_dir / "rotation_consistency.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\n  📈 Rotation consistency plot saved: {save_path}")


# ==================== 主流程 ====================

def evaluate_unit(unit_path: Path, args, visualizer: Visualizer = None, collector: MetricsCollector = None) -> EvaluationMetrics:
    """评估单个序列"""
    from StandardAdapter import StandardAdapter

    # 加载数据
    unit = StandardAdapter.load(unit_path)

    # 确定层级信息
    dataset = args.dataset or ""
    group = args.group or ""

    # 获取时间范围参数
    time_range = args.time_range if args.time_range != (None, None) else None

    # 创建评估器并评估
    evaluator = StandardEvaluator(
        unit, dataset=dataset, group=group,
        unit_path=unit_path,
        save_plots=(args.visual if (visualizer and unit_path) else False),
        z_axis_up=args.z_axis_up,
        time_range=time_range,
        enable_calibration=args.enable_calibration
    )
    metrics, debug_data = evaluator.evaluate()

    # 可视化
    if visualizer:
        if args.visual:
            # 时间对齐图
            visualizer.plot_time_alignment(
                unit.name,
                debug_data['i_gyro_raw'],
                debug_data['i_gyro_aligned'],
                debug_data['g_gyro'],
                debug_data['time_shift_ms'],
                debug_data.get('t_us'),
                unit_path  # 传入路径以支持保存到序列目录
            )
            # 2D 轨迹图
            visualizer.plot_trajectory_2d(unit.name, unit.gt_data, unit_path=unit_path)
            # 加速度旋转对比图
            if 'i_acce_synced' in debug_data and 'acc_w' in debug_data:
                visualizer.plot_acceleration_rotation(
                    unit.name,
                    debug_data['i_acce_synced'],
                    debug_data['acc_w'],
                    debug_data.get('t_us'),
                    unit_path
                )

        if args.rerun:
            visualizer.launch_rerun(unit.name, unit)

    # 添加到收集器
    if collector:
        collector.add(metrics, unit_path, unit)
    else:
        # 如果没有收集器，打印简要结果
        print(f"\n📊 {metrics.unit_name}: RMSE {metrics.raw_rmse:.4f} → {metrics.final_rmse:.4f} rad/s")

    return metrics


def evaluate_group(group_path: Path, args, visualizer: Visualizer = None, collector: MetricsCollector = None, recursive: bool = False) -> List[EvaluationMetrics]:
    """评估数据组

    Args:
        group_path: 数据组路径
        args: 参数
        visualizer: 可视化器
        collector: 指标收集器
        recursive: 是否递归搜索子目录
    """
    group_path = Path(group_path)
    metrics_list = []

    # 查找所有单元
    if recursive:
        # 递归搜索：查找所有包含 imu.csv 和 (gt.csv 或 rtab.csv) 的子目录
        imu_dirs = {d.parent for d in group_path.rglob("imu.csv") if d.is_file()}
        gt_dirs = {d.parent for d in group_path.rglob("gt.csv") if d.is_file()}
        rtab_dirs = {d.parent for d in group_path.rglob("rtab.csv") if d.is_file()}
        unit_dirs = sorted(imu_dirs & (gt_dirs | rtab_dirs))
        print(f"\n📁 Found {len(unit_dirs)} units in {group_path.name} (recursive search)")
    else:
        # 非递归：只搜索直接子目录
        unit_dirs = []
        for d in group_path.iterdir():
            if d.is_dir() and (d / "imu.csv").exists():
                if (d / "gt.csv").exists() or (d / "rtab.csv").exists():
                    unit_dirs.append(d)
        print(f"\n📁 Found {len(unit_dirs)} units in {group_path.name}")
    for unit_dir in sorted(unit_dirs):
        print(f"  → Evaluating: {unit_dir.name}")
        try:
            metrics = evaluate_unit(unit_dir, args, visualizer, collector)
            metrics_list.append(metrics)
        except Exception as e:
            print(f"  ❌ Failed: {e}")

    return metrics_list


def evaluate_dataset(dataset_path: Path, args, visualizer: Visualizer = None, collector: MetricsCollector = None, recursive: bool = False) -> List[EvaluationMetrics]:
    """评估数据集

    Args:
        dataset_path: 数据集路径
        args: 参数
        visualizer: 可视化器
        collector: 指标收集器
        recursive: 是否递归搜索子目录
    """
    dataset_path = Path(dataset_path)
    metrics_list = []

    # 查找所有组
    group_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    print(f"\n📂 Found {len(group_dirs)} groups in {dataset_path.name}")
    for group_dir in sorted(group_dirs):
        print(f"\n📁 Group: {group_dir.name}")
        group_metrics = evaluate_group(group_dir, args, visualizer, collector, recursive=recursive)
        metrics_list.extend(group_metrics)

    return metrics_list


def main():
    """主入口"""
    # 解析参数
    parser = EvaluatorArgsParser()
    args = parser.parse()

    # 确定评估路径
    if args.unit:
        eval_path = Path(args.unit)
        eval_type = "unit"
    elif args.group:
        eval_path = Path(args.group)
        eval_type = "group"
    elif args.dataset:
        eval_path = Path(args.dataset)
        eval_type = "dataset"
    else:
        print("❌ Error: Please specify --dataset, --group, or --unit")
        return

    # 创建指标收集器（默认保存到序列目录）
    collector = MetricsCollector()

    # 创建可视化器（默认保存到序列目录）
    visualizer = Visualizer(save_dir=Path("."), save_to_unit_dir=True) if (args.visual or args.rerun) else None

    # 执行评估
    print(f"\n{'='*80}")
    print(f"🚀 Starting Evaluation: {eval_path}")
    print(f"{'='*80}")

    if eval_type == "unit":
        metrics = evaluate_unit(eval_path, args, visualizer, collector)
        metrics_list = [metrics]
    elif eval_type == "group":
        metrics_list = evaluate_group(eval_path, args, visualizer, collector, recursive=args.recursive)
    else:  # dataset
        metrics_list = evaluate_dataset(eval_path, args, visualizer, collector, recursive=args.recursive)

    # 打印摘要
    if metrics_list:
        collector.print_summary()

        # 旋转一致性分析（保存在评估目录下）
        if len(metrics_list) >= 2:
            collector.check_rotation_consistency(save_dir=eval_path)
    else:
        print("❌ No metrics collected")


if __name__ == "__main__":
    main()
