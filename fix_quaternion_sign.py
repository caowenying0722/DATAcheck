#!/usr/bin/env python3
"""
修复四元数符号翻转问题
确保相邻四元数之间的点积为正（符号一致）

用法: uv run python fix_quaternion_sign.py -i "imudata/changed/OXIOD/Oxford_Formatted/data1/seq1/gt.csv"
"""
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

def fix_quaternion_sign(quats_wxyz: np.ndarray) -> np.ndarray:
    """
    标准化四元数符号，确保相邻四元数符号一致

    原理：四元数q和-q表示相同的旋转，但为了连续性，
    我们需要确保相邻四元数之间的点积为正

    Args:
        quats_wxyz: 四元数数组 (N, 4)，格式为[w, x, y, z]

    Returns:
        修复后的四元数数组
    """
    quats_fixed = quats_wxyz.copy()

    # 从第二个四元数开始，检查与前一个的点积
    for i in range(1, len(quats_fixed)):
        q_prev = quats_fixed[i-1]
        q_curr = quats_fixed[i]

        # 计算点积
        dot_product = np.dot(q_prev, q_curr)

        # 如果点积为负，翻转当前四元数的符号
        if dot_product < 0:
            quats_fixed[i] = -q_curr

    return quats_fixed

def main():
    parser = argparse.ArgumentParser(description="Fix quaternion sign flips in GT files")
    parser.add_argument("-i", "--input", type=str, required=True, help="GT CSV file path")
    args = parser.parse_args()

    gt_path = Path(args.input)

    if not gt_path.exists():
        print(f"Error: File {gt_path} does not exist.")
        return

    print(f"Processing: {gt_path}")

    # 读取GT文件
    df = pd.read_csv(gt_path)

    # 提取四元数列 (q_RS_w, q_RS_x, q_RS_y, q_RS_z)
    quat_cols = [col for col in df.columns if col.startswith('q_RS_')]
    if len(quat_cols) != 4:
        print(f"Error: Expected 4 quaternion columns, found {len(quat_cols)}")
        return

    # 获取四元数数据
    quats = df[quat_cols].values

    print(f"  Original shape: {quats.shape}")
    print(f"  Format: {quat_cols}")

    # 检测符号翻转
    flips_before = 0
    for i in range(1, len(quats)):
        dot_product = np.dot(quats[i-1], quats[i])
        if dot_product < 0:
            flips_before += 1

    print(f"  Detected {flips_before} sign flips")

    # 修复符号
    quats_fixed = fix_quaternion_sign(quats)

    # 验证修复
    flips_after = 0
    for i in range(1, len(quats_fixed)):
        dot_product = np.dot(quats_fixed[i-1], quats_fixed[i])
        if dot_product < 0:
            flips_after += 1

    print(f"  After fix: {flips_after} sign flips (should be 0)")

    # 更新DataFrame
    df[quat_cols] = quats_fixed

    # 备份原文件
    backup_path = gt_path.with_suffix('.csv.backup')
    if not backup_path.exists():
        import shutil
        shutil.copy2(gt_path, backup_path)
        print(f"  Backup saved: {backup_path}")

    # 保存修复后的文件
    df.to_csv(gt_path, index=False)
    print(f"  Fixed file saved: {gt_path}")

if __name__ == "__main__":
    main()
