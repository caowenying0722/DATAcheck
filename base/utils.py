import numpy as np
from numpy.typing import NDArray


def angle_with_x_axis(vector: NDArray) -> float:
    """
    计算向量与x轴的夹角,遵循右手螺旋定律

    Parameters:
    -----------
    vector : NDArray
        二维向量 [x, y] 或三维向量 [x, y, z]
        对于三维向量,忽略z分量,在xy平面计算

    Returns:
    --------
    float
        与x轴的夹角(度),范围 -180 到 180
        - 正值: 逆时针方向(右手螺旋定则)
        - 负值: 顺时针方向

    Examples:
    ---------
    >>> angle_with_x_axis([1, 0])      # x轴正方向
    0.0
    >>> angle_with_x_axis([0, 1])      # y轴正方向
    90.0
    >>> angle_with_x_axis([-1, 0])     # x轴负方向
    180.0
    >>> angle_with_x_axis([0, -1])     # y轴负方向
    -90.0
    """
    vector = np.asarray(vector)

    # 提取x和y分量
    if len(vector) >= 2:
        x, y = vector[0], vector[1]
    else:
        raise ValueError("向量维度必须至少为2维")

    # 使用arctan2计算角度,范围是[-pi, pi]
    angle_rad = np.arctan2(y, x)

    # 转换为度数
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def angle_between_vectors(v1: NDArray, v2: NDArray) -> float:
    """
    计算两个向量之间的夹角,遵循右手螺旋定律

    Parameters:
    -----------
    v1 : NDArray
        第一个向量 [x, y] 或 [x, y, z]
    v2 : NDArray
        第二个向量 [x, y] 或 [x, y, z]

    Returns:
    --------
    float
        从v1到v2的夹角(度),范围 -180 到 180
        - 正值: v1逆时针旋转到v2
        - 负值: v1顺时针旋转到v2

    Examples:
    ---------
    >>> angle_between_vectors([1, 0], [0, 1])      # x轴到y轴
    90.0
    >>> angle_between_vectors([1, 0], [0, -1])     # x轴到y轴负方向
    -90.0
    """
    # 计算每个向量与x轴的夹角
    angle1 = angle_with_x_axis(v1)
    angle2 = angle_with_x_axis(v2)

    # 计算相对角度
    diff = angle2 - angle1

    # 归一化到 [-180, 180]
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360

    return diff
