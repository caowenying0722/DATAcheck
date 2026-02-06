# 可视化工具使用指南

`visualize_rotation.py` 是IMU-GT旋转对齐可视化工具，可以生成多种图表帮助分析校准效果。

## 功能

1. **旋转对比图** (`rotation_comparison.png`) - 4个子图
   - 角速度模长对比
   - X分量对比
   - Y分量对比
   - Z分量对比（新增）

2. **时间对齐图** (`time_alignment.png`) - 3个子图
   - 对齐前IMU vs GT
   - 对齐后IMU vs GT
   - 对齐效果差值

3. **GT轨迹图** (`gt_trajectory.png`) - 4个子图
   - 3D轨迹
   - 位置分量
   - 欧拉角
   - 速度分量

4. **校准汇总图** (`calibration_summary.png`) - 柱状图
   - Raw vs Final RMSE对比

## 使用方法

### 可视化脚本基本用法

```bash
# 单个序列，使用全部数据
python visualize_rotation.py -u /path/to/unit

# 单个序列，指定时间范围（秒）
python visualize_rotation.py -u /path/to/unit --time-start 10 --time-end 50

# 跳过GT轨迹图（加速）
python visualize_rotation.py -u /path/to/unit --no-gt

# 批量处理整个数据集
python visualize_rotation.py -d /path/to/dataset

# 批量处理 + 时间范围
python visualize_rotation.py -d /path/to/dataset --time-start 0 --time-end 100
```

### 评估脚本（支持时间范围）

```bash
# 评估单个序列，使用全部数据
python StandardEvaluator.py -u /path/to/unit

# 评估单个序列，指定时间范围（秒）
python StandardEvaluator.py -u /path/to/unit -t 10 50

# 批量评估，使用时间范围
python StandardEvaluator.py -d /path/to/dataset -t 0 100

# 评估 + 可视化（带时间范围）
python StandardEvaluator.py -u /path/to/unit -t 10 50 -v

# 使用数据集校准 + 时间范围
python StandardEvaluator.py -u /path/to/unit -t 10 50 --use-dataset-calib
```

**注意**：时间范围参数在两个脚本中的格式不同
- `visualize_rotation.py`: `--time-start 10 --time-end 50`
- `StandardEvaluator.py`: `-t 10 50` （更简洁）

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `-u, --unit` | 序列路径 | `-u imudata/RoNIN/Data/ch1/a001_2` |
| `-d, --dataset` | 数据集路径（批量） | `-d imudata/RoNIN/Data/ch1` |
| `--time-start` | 开始时间（秒） | `--time-start 10` |
| `--time-end` | 结束时间（秒） | `--time-end 50` |
| `--no-gt` | 跳过GT轨迹图 | `--no-gt` |

## 输出说明

### 旋转对比图 (4个子图)

```
┌─────────────────────────────────────┐
│      Gyroscope Norm Comparison      │
│  无校准、数据集校准、自动校准 vs GT  │
├─────────────────────────────────────┤
│        Gyroscope X Component        │
│     X轴角速度各方法对比 vs GT        │
├─────────────────────────────────────┤
│        Gyroscope Y Component        │
│     Y轴角速度各方法对比 vs GT        │
├─────────────────────────────────────┤
│        Gyroscope Z Component        │
│     Z轴角速度各方法对比 vs GT        │
└─────────────────────────────────────┘
```

**如何判断校准效果：**
- 观察哪种校准方法的曲线与GT（黑色虚线）最接近
- 曲线重合度高 = 校准效果好
- 注意符号、幅值、相位是否一致

### 校准汇总图

如果未找到 `evaluation.txt`，会生成占位图并提示：

```
⚠️  未找到评估报告 (evaluation.txt)
请先运行评估:
  python StandardEvaluator.py -u /path/to/unit
```

## 实用技巧

### 1. 快速浏览数据

```bash
# 只看旋转对比和时间对齐，跳过GT轨迹
python visualize_rotation.py -u /path/to/unit --no-gt
```

### 2. 分析特定时段

```bash
# 只查看前10秒
python visualize_rotation.py -u /path/to/unit --time-start 0 --time-end 10

# 只查看中间段
python visualize_rotation.py -u /path/to/unit --time-start 30 --time-end 60
```

### 3. 批量对比分析

```bash
# 对比整个数据集
python visualize_rotation.py -d imudata/RoNIN/Data/test

# 然后检查每个序列的图表，找出校准效果差的
```

## 常见问题

### Q: 图表中文字显示为方块？

A: 中文字体缺失警告，不影响功能。图表仍能正常生成。

### Q: 校准汇总图为空？

A: 需要先运行评估：
```bash
python StandardEvaluator.py -u /path/to/unit
```

### Q: 如何判断哪种校准方式最优？

A: 在旋转对比图中：
1. 找到与GT曲线最接近的那条
2. 注意模长、X、Y、Z三个分量都要接近
3. 自动校准（红色）通常是最优的

## 示例输出

```bash
$ python visualize_rotation.py -u imudata/RoNIN/Data/ch1/a001_2 --time-start 10 --time-end 50

=== 旋转对齐可视化: a001_2 ===
  时间范围: 10s - 50s (8000 点)
  📈 旋转对比图保存: imudata/RoNIN/Data/ch1/a001_2/rotation_comparison.png

=== 时间对齐可视化: a001_2 ===
  时间范围: 10s - 50s
  📈 时间对齐图保存: imudata/RoNIN/Data/ch1/a001_2/time_alignment.png

=== 校准效果汇总: a001_2 ===
  📊 汇总图保存: imudata/RoNIN/Data/ch1/a001_2/calibration_summary.png
```

## 相关文档

- [校准使用指南](CALIBRATION.md)
- [快速参考](QUICKSTART.md)
- [坐标系说明](COORDINATE_SYSTEM.md)
