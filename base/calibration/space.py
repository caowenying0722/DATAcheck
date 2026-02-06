

import numpy as np
from scipy.spatial.transform import Rotation

from base.interpolate import get_time_series

from ..datatype import Pose, PosesData


def get_angle(v1, v2):
    # v1 = v1 / np.linalg.norm(v1)
    # v2 = v2 / np.linalg.norm(v2)
    # 计算夹角
    dot = np.dot(v1, v2)
    det = np.cross(v1, v2)
    return np.arctan2(det, dot)


def global12(
    cs1: PosesData,
    cs2: PosesData,
) -> Pose:
    """calibrateHandEye return R_gc, t_gc
    隐含信息：base 和 target 为刚体，gripper 和 camera 为刚体。
    """
    # 插值对齐
    t_new_us = get_time_series([cs1.t_us, cs2.t_us], rate=100)
    t_new_us = t_new_us[: 100 * 20]
    cs1 = cs1.interpolate(t_new_us)
    cs2 = cs2.interpolate(t_new_us)

    angles = []
    for i in range(len(cs1)):
        v1 = (cs1.ps[i] - cs1.ps[0])[:2]
        v2 = (cs2.ps[i] - cs2.ps[0])[:2]

        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        if l1 > 2 and np.abs(l1 - l2) < 3:
            ang = get_angle(v1, v2)
            angles.append(ang)

    # 构造绕z轴的旋转
    rad = -np.mean(np.array(angles))
    rot12 = Rotation.from_rotvec([0, 0, rad])
    return Pose(rot12, np.zeros(3))
