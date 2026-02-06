import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def slerp_rotation(rots: Rotation, t_old_us: NDArray, t_new_us: NDArray) -> Rotation:
    assert len(rots) == len(t_old_us)
    slerp = Slerp(t_old_us, rots)
    rots_new = slerp(t_new_us)
    return rots_new


def interpolate_vector3(
    vec3: NDArray,
    t_old_us: NDArray,
    t_new_us: NDArray,
    bounds_error: bool = False,
) -> NDArray:
    assert len(vec3) == len(t_old_us)
    vec3 = np.array(vec3)

    interp = interp1d(
        t_old_us,
        vec3,
        axis=0,
        kind="cubic",
        bounds_error=bounds_error,
        fill_value="extrapolate",
    )
    vec3_new = interp(t_new_us)
    return vec3_new


def get_time_series(
    ts_us: list[NDArray],
    t_start_s: int | None = None,
    t_end_s: int | None = None,
    *,
    rate: float = 100.0,
) -> NDArray:
    t_start_us = max([t[0] for t in ts_us])
    t_end_us = min([t[-1] for t in ts_us])
    interval = 1e6 / rate
    assert t_start_us < t_end_us, (
        "Time series must be non-empty and have a valid interval"
    )
    t_us = np.arange(t_start_us, t_end_us, interval, dtype=np.int64)
    # 限制时间轴长度
    start_idx = 0 if t_start_s is None else int(max(t_start_s * rate, 0))
    end_idx = len(t_us) if t_end_s is None else int(min(t_end_s * rate, len(t_us)))
    t_us = t_us[start_idx:end_idx]

    assert t_us[0] >= t_start_us and t_us[-1] <= t_end_us, (
        "t_us must be within t_start_us and t_end_us"
    )
    return t_us
