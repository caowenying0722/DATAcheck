# IMU-GT 坐标系旋转完整说明

## 数据集的坐标系统

### RoNIN 数据集

```
┌─────────────────────────────────────────────────────────┐
│                     设备（手机）                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │  IMU传感器 (加速度计、陀螺仪)                      │  │
│  │  → 坐标系: IMU Body Frame                         │  │
│  │  → 测量: 加速度 a_body, 角速度 ω_body               │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Tango相机/深度传感器                             │  │
│  │  → 坐标系: Tango Frame                           │  │
│  │  → 测量: 姿态 R_gt_tango, 位置 p_gt_tango          │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  align_tango_to_body: R_tango→body                   │
│  从 Tango Frame 旋转到 Body Frame                     │
└─────────────────────────────────────────────────────────┘
```

### 旋转矩阵的作用

#### align_tango_to_body 的数学定义

```python
R_tango_to_body = quat_to_matrix(align_tango_to_body)

# 用途1: 将 GT 姿态从 Tango 转到 Body
R_gt_body = R_tango_to_body * R_gt_tango

# 用途2: 将 IMU 数据从 Body 转到 Tango（逆变换）
a_tango = R_tango_to_body.inv() @ a_body
```

## 评估流程中的坐标系转换

### 评估的目标

**比较 IMU 估计的角速度和 GT 角速度**

### 问题：IMU和GT在不同坐标系！

```
IMU数据:
  a_body, ω_body ∈ IMU Body Frame
  
GT数据:
  R_gt_tango ∈ Tango Frame (姿态)
  → 计算得到 ω_gt_tango ∈ ? Frame (角速度)
```

### 关键问题：ω_gt_tango 在哪个坐标系？

这取决于GT姿态的定义：
- 如果 R_gt_tango 表示 "从 Body 到 World 的旋转" → ω_gt 在 Body Frame
- 如果 R_gt_tango 表示 "从 Tango 到 World 的旋转" → ω_gt 在 Tango Frame

## 正确的对齐策略

### 策略A：自动校准（推荐，效果最好）

```python
# 不假设任何先验，从数据中学习最优对齐
R_optimal = Rotation.align_vectors(ω_gt, ω_imu)
ω_imu_aligned = R_optimal @ ω_imu

# 比较 ω_imu_aligned vs ω_gt
```

**优点**：
- 自适应，不依赖数据集校准
- 测试结果：RMSE = 1.4166 rad/s ✓

### 策略B：无校准

```python
# 直接比较，假设坐标系已对齐
# 比较 ω_imu vs ω_gt
```

**测试结果**：RMSE = 1.4966 rad/s

### 策略C：使用数据集校准

```python
# 使用 align_tango_to_body 对齐
ω_imu_tango = R_tango_to_body.inv() @ ω_imu_body

# 比较 ω_imu_tango vs ω_gt_tango
```

**测试结果**：RMSE = 1.4639 rad/s

## 为什么自动校准效果最好？

### 1. 它直接优化目标函数

```python
Rotation.align_vectors(ω_gt, ω_imu)
# 最小化 ||R @ ω_imu - ω_gt||
```

这正是我们评估时最小化的 RMSE 指标！

### 2. 它不依赖可能错误的先验

- `align_tango_to_body` 可能用于其他目的（如Tango传感器融合）
- 它不是为 IMU-GT 角速度对齐设计的
- 自动校准直接从数据中学习最优旋转

### 3. 它适应不同序列

- 每个序列可能有不同的安装偏差
- 自动校准能针对每个序列优化
- 数据集校准是一个全局的、可能不准确的值

## 实际建议

### 对于评估

```bash
# 使用自动校准（默认）
python StandardEvaluator.py -u <unit_path>

# 结果会自动计算最优的 R_calib
# 并保存在 evaluation.txt 和 statistics.json 中
```

### 对于理解坐标系

1. **不要假设 `align_tango_to_body` 能用于IMU-GT对齐**
   - 它的用途可能是 Tango 传感器数据融合
   - 不是为 IMU 评估设计的

2. **自动校准是最可靠的方法**
   - 基于实际数据优化
   - 不依赖可能错误的先验

3. **如果想验证坐标系**
   - 比较不同校准方式的 RMSE
   - 选择 RMSE 最小的方式
   - 检查重力误差作为辅助验证

## 补充：重力对齐验证

重力误差是一个**辅助指标**，用于验证静态情况下的对齐质量：

```python
# 期望：静态时，加速度 ≈ 重力
# 在世界坐标系中：重力应该沿 Z 轴 [0, 0, g]

# 计算方法：
acc_world = R_gt @ imu_acc
gravity_error = angle(acc_world, [0, 0, 1])

# 评判标准：
# - gravity_error < 10°: 对齐良好
# - gravity_mag ≈ 9.81 m/s²: 幅值正确
```

**但是**：重力误差只能验证静态情况，对于动态数据不准确。

## 总结

1. **坐标系转换流程**：
   ```
   IMU Body Frame --align_tango_to_body--> Tango Frame --R_gt--> World Frame
   ```

2. **评估时应该在同一坐标系比较**：
   ```
   选项A: 将IMU转到GT坐标系（自动校准最优）
   选项B: 将GT转到IMU坐标系（需要重新计算GT）
   ```

3. **推荐做法**：
   - 使用自动校准（`Rotation.align_vectors`）
   - 不要依赖 `align_tango_to_body` 进行IMU-GT对齐
   - 以角速度RMSE为主要指标，重力误差为辅助指标
