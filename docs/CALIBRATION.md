# RoNIN/TLIO 数据集校准使用指南

## 重要发现

### ⚠️ RoNIN 数据集警告

**对于 RoNIN 数据集，不建议使用 `--use-dataset-calib` 参数！**

#### 测试结果对比

| 校准方式 | Gravity Error | Gravity Magnitude | 说明 |
|---------|--------------|-------------------|------|
| **无校准 (Raw)** | **5.15°** | 6.10 m/s² | IMU和GT自然对齐 |
| 自动校准 | 54.11° | 2.64 m/s² | 自动计算的校准 |
| **数据集校准** | **85.07°** ❌ | 2.13 m/s² | 使用align_tango_to_body |

#### 为什么会这样？

RoNIN 数据集的 `info.json` 中包含 `align_tango_to_body` 四元数，但这个旋转**不是用来做 IMU-GT 对齐的**！

根据代码注释（UniversalConverter.py:1068-1072）：
> NOTE: align_tango_to_body is NOT applied here
> Reason: Based on testing, applying align_tango_to_body does NOT correctly align
> GT orientation to IMU body frame (results in 98° error from gravity)

`align_tango_to_body` 的真实用途可能是：
- Tango 相机/深度传感器的坐标系转换
- 用于数据集可视化的对齐
- **不是用于 IMU 评估的校准**

### ✅ TLIO 数据集

TLIO 数据集的校准可能更可靠，但仍需验证。

## 推荐用法

### 1. RoNIN 数据集（推荐）

```bash
# 使用自动校准（默认）
python StandardEvaluator.py -u <unit_path>

# 或者不使用任何空间校准
# （评估器会自动计算最优校准）
```

### 2. TLIO 数据集

```bash
# 可以尝试数据集校准，但需要验证结果
python StandardEvaluator.py -u <unit_path> --use-dataset-calib

# 对比结果
python StandardEvaluator.py -u <unit_path>
```

### 3. 如何判断校准是否有效？

查看评估报告中的关键指标：
- **Gravity Error**: 应该 < 10°（越小越好）
- **Gravity Magnitude**: 应该接近 9.81 m/s²
- **RMSE**: 校准后应该降低

如果使用 `--use-dataset-calib` 后：
- Gravity Error 显著增大（如从 5° 增到 85°）
- Gravity Magnitude 偏离 9.81 m/s²

**说明数据集校准不适用，请使用默认的自动校准！**

## 校准逻辑说明

### 坐标系

```
IMU Body Frame: IMU传感器物理坐标系
  ↓ (align_tango_to_body ?)
Tango Frame: Tango相机/深度传感器坐标系
  ↓
World Frame: 全局参考坐标系（Z轴与重力对齐）
```

### align_tango_to_body 的含义

- **数学含义**: 从 Tango 坐标系旋转到 Body 坐标系
- **实际用途**: 可能用于 Tango 传感器的数据融合
- **错误用法**: 直接用于 IMU-GT 旋转对齐

### 自动校准 vs 数据集校准

#### 自动校准（推荐）
```python
# 基于角速度向量对齐
R_calib, _ = Rotation.align_vectors(gt_gyro, imu_gyro)
```
- 优点：自适应，能找到最优旋转
- 缺点：依赖 GT 角速度的准确性

#### 数据集校准（谨慎使用）
```python
R_calib = Rotation.from_quat(align_tango_to_body)
```
- 优点：使用数据集提供的先验信息
- 缺点：可能不适用于评估任务

## 常见问题

### Q: 为什么 Raw 的重力误差反而很小？

A: RoNIN 数据集在采集时可能已经做了坐标系对齐，所以 IMU 和 GT 自然对齐。额外的校准反而会破坏这种对齐。

### Q: 什么时候应该使用数据集校准？

A: 只有在以下情况才考虑：
1. 自动校准的 Gravity Error > 20°
2. 数据集文档明确说明校准的用途
3. 验证使用后指标确实改善

### Q: 如何验证哪种校准方式更好？

A: 对比两种方式的结果：
```bash
# 自动校准
python StandardEvaluator.py -u <unit_path>

# 数据集校准
python StandardEvaluator.py -u <unit_path> --use-dataset-calib

# 对比 evaluation.txt 中的:
# - Gravity Error (越小越好)
# - Gravity Magnitude (越接近9.81越好)
# - Final RMSE (越小越好)

# 或者使用可视化工具直观对比
python visualize_rotation.py -u <unit_path>
```

## 可视化工具

使用 `visualize_rotation.py` 可以直观对比不同校准方式的效果：

```bash
# 单个序列
python visualize_rotation.py -u /path/to/unit

# 批量处理
python visualize_rotation.py -d /path/to/dataset
```

### 生成的可视化图表

| 图表 | 说明 | 文件名 |
|-----|------|--------|
| **旋转对比图** | 比较不同校准方式下的角速度（模长、X分量、Y分量） | `rotation_comparison.png` |
| **时间对齐图** | 展示时间对齐前后的效果 | `time_alignment.png` |
| **GT轨迹图** | 3D轨迹、位置分量、姿态欧拉角、速度分量 | `gt_trajectory.png` |
| **校准汇总图** | Raw vs Final RMSE 对比柱状图 | `calibration_summary.png` |

### 可视化示例

**旋转对比图**包含以下曲线：
- 无校准（原始IMU数据）
- 数据集校准（使用 `align_tango_to_body`）
- 数据集校准（逆变换）
- 自动校准（`Rotation.align_vectors`）
- Ground Truth（参考基准）

通过观察不同曲线与 GT 的贴合程度，可以直观判断哪种校准方式最优。

## 修改建议

如果发现某些序列确实需要数据集校准（经验证后改善指标），可以考虑：

1. **添加白名单机制**：为需要数据集校准的序列创建白名单
2. **自动验证机制**：自动对比两种校准方式，选择更好的
3. **数据集特定处理**：针对不同数据集使用不同的默认策略

## 参考文档

- UniversalConverter.py:1068-1072 - RoNIN 转换器的说明注释
- StandardEvaluator.py:344-433 - 评估和校准逻辑
- 测试结果：/tmp/test_calib_direction.py
