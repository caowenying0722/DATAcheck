import numpy as np
from numpy._typing import NDArray
from scipy.spatial.transform import Rotation

from base.datatype import PosesData
from base.interpolate import get_time_series


def _get_angvels(t_us: NDArray, rots: Rotation, step: int = 1):
    """获取角速度列表"""
    n = len(rots)
    step = max(int(step), 1)
    assert n >= 2, "At least two rotations are required"

    As: list = []
    Ts = []
    for i in range(0, n - step, step):
        drot = rots[i].inv() * rots[i + step]
        angle = float(np.linalg.norm(drot.as_rotvec()))
        dt_s = (t_us[i + step] - t_us[i]) * 1e-6
        assert dt_s > 0, "Time difference must be positive"
        ang_vel = angle / dt_s
        As.append(ang_vel)
        Ts.append(t_us[i])
    return As, Ts


def match21(
    cs1: PosesData,
    cs2: PosesData,
    *,
    time_range=(0, 50),
    resolution=100,
) -> int:
    # 分辨率不能大于时间序列的采样率，否则没有插值的意义
    rate = min(cs1.rate, cs2.rate)
    resolution = min(resolution, rate)
    print(f"Rate1:{cs1.rate}, Rate2: {cs2.rate}, reso: {resolution}")

    t_new_us = get_time_series([cs1.t_us, cs2.t_us], *time_range, rate=rate)
    cs1 = cs1.interpolate(t_new_us)
    cs2 = cs2.interpolate(t_new_us)
    print(f"使用时间范围：{(cs1.t_us[-1] - cs1.t_us[0]) / 1e6} 秒, 数量 {len(cs1)}")

    # 获取原始角速度序列 (list)
    seq1_list, t1 = _get_angvels(cs1.t_us, cs1.rots, step=int(rate / resolution))
    seq2_list, t2 = _get_angvels(cs2.t_us, cs2.rots, step=int(rate / resolution))
    t_new_us = t1

    # === 【关键修改开始】 ===
    # 1. 转换为 Numpy 数组以便处理
    arr1 = np.array(seq1_list)
    arr2 = np.array(seq2_list)

    # 2. 去尖峰 (Despiking/Clipping)
    # 正常手持运动角速度通常 < 10 rad/s。
    # 这里设置 15.0 作为安全阈值，超过的全切掉，防止 Vicon 噪声主导 Correlation。
    LIMIT = 15.0
    arr1 = np.clip(arr1, 0, LIMIT)
    arr2 = np.clip(arr2, 0, LIMIT)

    # 3. 去均值 (Zero-Normalized)
    # 互相关最好在去均值后进行，这样关注的是"波动的形状"而不是"数值的大小"
    arr1 -= np.mean(arr1)
    arr2 -= np.mean(arr2)
    # === 【关键修改结束】 ===

    # 使用处理后的数组进行互相关计算
    corr = np.correlate(arr1, arr2, mode="full")
    lag_arr = np.arange(-len(arr2) + 1, len(arr1))
    lag = lag_arr[np.argmax(corr)]
    
    # 注意：t_new_us 是基于 _get_angvels 返回的时间轴
    dt = t_new_us[1] - t_new_us[0]
    t21_us = lag * dt
    
    print("Ground time gap: ", t21_us / 1e6)

    # 下面的逻辑似乎是想验证一下或者截取，保持原样
    # 但注意：这里修改 cs2 可能会影响外部传入的对象引用，不过 float += 是安全的
    # time_range = (0, 20)
    # cs1 = cs1.get_time_range(time_range)
    # cs2 = cs2.get_time_range(time_range)
    # cs2.t_us += t21_us

    return int(t21_us)

# def match21(
#     cs1: PosesData,
#     cs2: PosesData,
#     *,
#     time_range=(0, 50),
#     resolution=100,
# ) -> int:
#     # 分辨率不能大于时间序列的采样率，否则没有插值的意义
#     rate = min(cs1.rate, cs2.rate)
#     resolution = min(resolution, rate)
#     print(f"Rate1:{cs1.rate}, Rate2: {cs2.rate}, reso: {resolution}")

#     t_new_us = get_time_series([cs1.t_us, cs2.t_us], *time_range, rate=rate)
#     cs1 = cs1.interpolate(t_new_us)
#     cs2 = cs2.interpolate(t_new_us)
#     print(f"使用时间范围：{(cs1.t_us[-1] - cs1.t_us[0]) / 1e6} 秒, 数量 {len(cs1)}")

#     seq1, t1 = _get_angvels(cs1.t_us, cs1.rots, step=int(rate / resolution))
#     seq2, t2 = _get_angvels(cs2.t_us, cs2.rots, step=int(rate / resolution))
#     t_new_us = t1

#     corr = np.correlate(seq1, seq2, mode="full")
#     lag_arr = np.arange(-len(seq2) + 1, len(seq1))
#     lag = lag_arr[np.argmax(corr)]
#     t21_us = lag * (t_new_us[1] - t_new_us[0])
#     print("Ground time gap: ", t21_us / 1e6)

#     time_range = (0, 20)
#     cs1 = cs1.get_time_range(time_range)
#     cs2 = cs2.get_time_range(time_range)
#     cs2.t_us += t21_us

#     return t21_us
