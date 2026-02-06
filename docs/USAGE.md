# ModelVerify 使用指南

> **文档更新日期：2026-01-08**

本文档提供了 ModelVerify 项目的详细使用方法和最佳实践。

## 目录

- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [命令行使用](#命令行使用)
- [Python API 使用](#python-api-使用)
- [数据处理流程](#数据处理流程)
- [模型推理](#模型推理)
- [结果评估](#结果评估)
- [可视化](#可视化)
- [高级用法](#高级用法)
- [常见问题](#常见问题)

---

## 环境配置

### 前置要求

- Python >= 3.11
- uv (推荐) 或 pip

### 安装步骤

```bash
# 克隆项目
git clone <repository-url>
cd ModelVerify

# 使用 uv 安装依赖（推荐）
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows

# 验证安装
python -c "import torch; import numpy; import rerun; print('安装成功！')"
```

### 目录结构准备

```bash
# 创建必要的目录
mkdir -p models datasets results
```

---

## 数据准备

### 数据单元结构

每个数据单元应包含以下文件：

```
unit_name/
├── imu.csv           # IMU传感器数据
├── gt.csv            # 真值位姿数据
├── cam.csv           # 相机数据（可选）
├── fusion.csv        # 融合数据（可选）
├── Calibration.json  # 标定文件
└── DataCheck.json    # 数据检查文件
```

### IMU 数据格式 (imu.csv)

```csv
timestamp,gyro_x,gyro_y,gyro_z,acce_x,acce_y,acce_z,qw,qx,qy,qz,time_offset,magn_x,magn_y,magn_z
```

字段说明：
- `timestamp`: 时间戳（微秒）
- `gyro_x/y/z`: 陀螺仪数据（rad/s）
- `acce_x/y/z`: 加速度计数据（m/s²）
- `qw/qx/qy/qz`: AHRS四元数（标量在前）
- `time_offset`: 时间偏移
- `magn_x/y/z`: 磁力计数据（可选）

### 真值数据格式 (gt.csv)

```csv
timestamp,tx,ty,tz,qw,qx,qy,qz
```

字段说明：
- `timestamp`: 时间戳（微秒）
- `tx/ty/tz`: 位置坐标（米）
- `qw/qx/qy/qz`: 姿态四元数（标量在前）

### 标定文件格式 (Calibration.json)

```json
[
  {
    "rot_sensor_gt": [[3x3 旋转矩阵]],
    "trans_sensor_gt": [x, y, z],
    "rot_ref_sensor_gt": [[3x3 旋转矩阵]],
    "trans_ref_sensor_gt": [x, y, z]
  }
]
```

---

## 命令行使用

### 基本命令

#### 1. 验证单个数据单元

```bash
# 使用默认模型
python main.py -u /path/to/unit

# 指定模型
python main.py -u /path/to/unit -m model1.pt model2.pt

# 指定模型路径
python main.py -u /path/to/unit -m model1.pt --models_path /custom/models/path

# 使用 AHRS 数据而非 GT 数据
python main.py -u /path/to/unit -m model1.pt --using_ahrs
```

#### 2. 验证整个数据集

```bash
# 验证数据集中的所有单元
python main.py -d /path/to/dataset -m model1.pt

# 指定时间范围（秒）
python main.py -u /path/to/unit -m model1.pt --time_range 10 100
```

### 参数说明

| 参数 | 简写 | 说明 | 必需 |
|------|------|------|------|
| `--unit` | `-u` | 单个数据单元路径 | 与 `-d` 二选一 |
| `--dataset` | `-d` | 数据集路径 | 与 `-u` 二选一 |
| `--models` | `-m` | 模型文件名（可指定多个） | 是 |
| `--models_path` | - | 模型文件夹路径 | 否（默认：models/） |
| `--using_ahrs` | - | 使用 AHRS 数据旋转 IMU | 否 |
| `--time_range` | - | 时间范围（秒） | 否 |

### 实用脚本

#### 数据对比可视化

```bash
# 可视化对比多个模型
python DrawCompare.py -u /path/to/unit -m model1.pt model2.pt

# 使用自定义融合算法
python DrawCompare.py -u /path/to/unit -m model1.pt --using_fusion
```

#### 模型转换

```bash
# 转换为 TorchScript
python TorchScript.py --model_path /path/to/model.py --output model_scripted.pt

# 转换为 Android
python TorchScript2Android.py --input model.pt --output model_android.pt
```

---

## Python API 使用

### 1. 基础用法

#### 加载单个数据并预测

```python
from base.datatype import UnitData
from base.model import ModelLoader, DataRunner, InertialNetworkData

# 加载数据
data = UnitData("/path/to/unit")
print(f"数据单元名称: {data.name}")
print(f"设备名称: {data.device_name}")
print(f"IMU 数据长度: {len(data.imu_data)}")
print(f"GT 数据长度: {len(data.gt_data)}")

# 加载模型
loader = ModelLoader("models")
model = loader.get_by_name("model1.pt")

# 创建数据处理器
processor = InertialNetworkData.set_step(20)

# 运行预测
runner = DataRunner(data, processor)
result = runner.predict(model)

print(f"预测完成，结果保存至: results/{data.name}/{model.name}.csv")
```

#### 批量预测多个模型

```python
from base.datatype import UnitData
from base.model import ModelLoader, DataRunner, InertialNetworkData

# 加载数据
data = UnitData("/path/to/unit")

# 加载多个模型
loader = ModelLoader("models")
models = loader.get_by_names(["model1.pt", "model2.pt", "model3.pt"])

# 设置处理器参数
DataClass = InertialNetworkData.set_step(20).set_rate(200)

# 批量预测
runner = DataRunner(data, DataClass)
results = runner.predict_batch(models)

for model, result in zip(models, results):
    print(f"{model.name}: 预测完成")
```

### 2. 数据集处理

#### 遍历整个数据集

```python
from base.datatype import DeviceDataset
from base.model import ModelLoader, DataRunner, InertialNetworkData

# 加载数据集
dataset = DeviceDataset("/path/to/dataset")
print(f"数据集包含 {len(dataset)} 个单元")

# 加载模型
loader = ModelLoader("models")
model = loader.get_by_name("model1.pt")

# 遍历所有单元
for unit_data in dataset:
    try:
        print(f"\n处理单元: {unit_data.name}")
        
        runner = DataRunner(unit_data, InertialNetworkData.set_step(20))
        runner.predict(model)
        
        print(f"✓ {unit_data.name} 完成")
    except Exception as e:
        print(f"✗ {unit_data.name} 失败: {e}")
```

### 3. 数据操作

#### 姿态操作

```python
from base.datatype import Pose
from scipy.spatial.transform import Rotation
import numpy as np

# 创建位姿
rot = Rotation.from_euler('xyz', [0, 0, 90], degrees=True)
trans = np.array([1.0, 2.0, 3.0])
pose = Pose(rot, trans, t_us=1000)

# 位姿组合
pose2 = Pose(Rotation.identity(), np.array([0, 0, 1]))
combined = pose.compose(pose2)

# 位姿求逆
inv_pose = pose.inverse()

# 获取偏航角
yaw_pose = pose.get_yaw_pose()
print(f"偏航角: {yaw_pose.rot.as_euler('ZXY')[0]}")
```

#### IMU 数据处理

```python
from base.datatype import ImuData

# 获取时间范围内的数据
imu_data = ImuData.from_csv("/path/to/imu.csv")
imu_data_range = imu_data.get_time_range((10.0, 20.0))  # 10-20秒

# 插值到新的时间序列
new_times = np.linspace(imu_data_range.t_us[0], imu_data_range.t_us[-1], 100)
interpolated = imu_data_range.interpolate(new_times)

# 变换到全局坐标系
global_imu = imu_data.transform()

# 统计信息
print(f"陀螺仪均值: {imu_data.gyro.mean(axis=0)}")
print(f"加速度计均值: {imu_data.acce.mean(axis=0)}")
```

### 4. 自定义模型使用

#### 使用 LSTM 模型

```python
from base.model import MemoryInertialNetwork

# LSTM 模型会自动检测（文件名包含 "lstm"）
loader = ModelLoader("models")
model = loader.get_by_name("lstm_model.pt")

# MemoryInertialNetwork 支持记忆状态管理
assert isinstance(model, MemoryInertialNetwork)
print(f"模型类型: {'LSTM' if 'lstm' in model.name else '普通网络'}")
```

#### 数据旋转增强

```python
from base.model import DataRunner, InertialNetworkData

# 运行预测
runner = DataRunner(data, InertialNetworkData.set_step(20))

# 使用不同旋转角度测试
degrees = [0, 45, 90, 135, 180]
results = runner.predict_rot(model, degrees=degrees)

for deg, result in zip(degrees, results):
    print(f"{deg}度旋转预测完成")
```

---

## 数据处理流程

### 完整的数据处理流程

```python
from base.datatype import UnitData, ImuData
from base.interpolate import get_time_series

# 1. 加载原始数据
data = UnitData("/path/to/unit")

# 2. 查看数据信息
print(f"IMU 采样率: {data.imu_data.rate:.2f} Hz")
print(f"GT 采样率: {data.gt_data.rate:.2f} Hz")
print(f"IMU 起始时间: {data.imu_data.t_us[0]} μs")
print(f"GT 起始时间: {data.gt_data.t_us[0]} μs")

# 3. 时间对齐
t_new_us = get_time_series([data.imu_data.t_us, data.gt_data.t_us])
imu_aligned = data.imu_data.interpolate(t_new_us)
gt_aligned = data.gt_data.interpolate(t_new_us)

print(f"对齐后数据长度: {len(t_new_us)}")

# 4. 空间变换
imu_global = imu_aligned.transform(gt_aligned.rots)

# 5. 数据预处理
# 移除重力
imu_global.acce -= np.array([0, 0, 9.81])

# 组合数据块
imu_block = np.hstack([imu_global.gyro, imu_global.acce])
print(f"数据块形状: {imu_block.shape}")
```

### 标定数据说明

```python
from base.datatype import CalibrationData, UnitData

data = UnitData("/path/to/unit")

# 标定数据会自动加载
print("标定信息:")
print(f"  传感器到GT的旋转:\n{data.calib_data.tf_sg_local.rot.as_matrix()}")
print(f"  传感器到GT的平移: {data.calib_data.tf_sg_local.p}")
print(f"  时间偏移: {data.check_data.t_gi_us} μs")
```

---

## 模型推理

### 1. 模型加载

```python
from base.model import ModelLoader

# 创建加载器
loader = ModelLoader("models")

# 列出所有可用模型
print("可用模型:")
for model_path in loader.models:
    print(f"  - {model_path.name}")

# 按名称加载单个模型
model = loader.get_by_name("model1.pt")

# 批量加载多个模型
models = loader.get_by_names(["model1.pt", "model2.pt"])

# 遍历所有模型
for model in loader:
    print(f"模型: {model.name}")
```

### 2. 推理流程

```python
from base.model import DataRunner, InertialNetworkData, ModelLoader
from base.datatype import UnitData

# 1. 加载数据和模型
data = UnitData("/path/to/unit")
loader = ModelLoader("models")
model = loader.get_by_name("model1.pt")

# 2. 配置数据处理器
# step=20: 每20个IMU数据点预测一次
# rate=200: IMU采样率为200Hz
DataClass = InertialNetworkData.set_step(20).set_rate(200)

# 3. 创建运行器
runner = DataRunner(
    data, 
    DataClass,
    time_range=(0, 100),  # 只处理0-100秒的数据
    using_gt=True         # 使用GT数据进行旋转
)

# 4. 运行预测
result = runner.predict(model)

# 5. 查看结果
print(f"预测点数: {len(result)}")
print(f"最终位置: {result.pose_list[-1].p}")
print(f"结果已保存至: results/{data.name}/{model.name}.csv")
```

### 3. 批处理优化

```python
from base.model import DataRunner, InertialNetworkData, ModelLoader
from base.datatype import DeviceDataset

# 加载数据集
dataset = DeviceDataset("/path/to/dataset")
loader = ModelLoader("models")
models = loader.get_by_names(["model1.pt", "model2.pt"])

# 配置处理器
DataClass = InertialNetworkData.set_step(20)

# 批量处理
for unit_data in dataset:
    runner = DataRunner(unit_data, DataClass)
    results = runner.predict_batch(models)
    
    print(f"单元 {unit_data.name} 完成")
    for model, result in zip(models, results):
        print(f"  {model.name}: {len(result)} 个预测点")
```

### 4. 处理结果

```python
import pandas as pd
from pathlib import Path

# 读取CSV结果
result_path = Path("results/unit_name/model1.csv")
df = pd.read_csv(result_path)

print("结果统计:")
print(df.describe())

# 提取位置和姿态
positions = df[['p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]']].values
quaternions = df[['q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []']].values

# 计算轨迹长度
from scipy.spatial.transform import Rotation

disps = np.diff(positions, axis=0)
trajectory_length = np.sum(np.linalg.norm(disps, axis=1))
print(f"轨迹长度: {trajectory_length:.2f} 米")
```

---

## 结果评估

### 1. 基础评估

```python
from base.datatype import UnitData, PosesData
from base.evaluate import Evaluation
import pandas as pd
from scipy.spatial.transform import Rotation

# 加载真值和预测结果
data = UnitData("/path/to/unit")

# 读取预测结果CSV
df = pd.read_csv(f"results/{data.name}/model1.csv")
t_us = df['#timestamp [us]'].values
ps = df[['p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]']].values
qs = df[['q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []']].values

# 创建预测位姿
pred_poses = PosesData(
    t_us=t_us,
    rots=Rotation.from_quat(qs.values, scalar_first=True),
    ps=ps
)

# 创建评估器
evaluator = Evaluation(data.gt_data)

# 计算评估指标
results = evaluator.get_eval(pred_poses, "model1")

# 打印结果
evaluator.print()

# 保存结果
evaluator.save(f"results/{data.name}/model1_eval.json")
```

### 2. 评估指标说明

项目计算以下评估指标：

#### APE (Absolute Pose Error) - 绝对位姿误差
- 描述：每个时间点上预测位姿与真值位姿之间的误差
- 包含位置误差和旋转误差
- 单位：米（位置）、弧度（旋转）

#### ATE (Absolute Trajectory Error) - 绝对轨迹误差
- 描述：预测轨迹与真值轨迹之间的平均距离误差
- 单位：米

#### RTE (Relative Pose Error) - 相对位姿误差
- 描述：固定时间间隔内的相对运动误差
- 默认间隔：60秒
- 单位：米（位置）、弧度（旋转）

### 3. 多模型对比

```python
from base.datatype import UnitData, PosesData
from base.evaluate import Evaluation
import pandas as pd
from scipy.spatial.transform import Rotation

data = UnitData("/path/to/unit")
evaluator = Evaluation(data.gt_data)

# 对比多个模型
model_names = ["model1.pt", "model2.pt", "model3.pt"]

for model_name in model_names:
    # 读取结果
    df = pd.read_csv(f"results/{data.name}/{model_name.replace('.pt', '')}.csv")
    
    # 创建位姿
    pred_poses = PosesData(
        t_us=df['#timestamp [us]'].values,
        rots=Rotation.from_quat(df[['q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []']].values, scalar_first=True),
        ps=df[['p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]']].values
    )
    
    # 评估
    evaluator.get_eval(pred_poses, model_name)

# 打印所有结果
evaluator.print()
evaluator.save(f"results/{data.name}/comparison_eval.json")
```

### 4. 可视化评估结果

```python
import matplotlib.pyplot as plt
import json

# 读取评估结果
with open("results/unit_name/comparison_eval.json") as f:
    results = json.load(f)

# 提取 ATE
models = []
ates = []
for key, value in results.items():
    if key.endswith("_ate_vec"):
        model_name = key.replace("_ate_vec", "")
        models.append(model_name)
        ates.append(value[0])  # x方向误差

# 绘制条形图
plt.figure(figsize=(10, 6))
plt.bar(models, ates)
plt.xlabel('模型')
plt.ylabel('ATE (米)')
plt.title('模型对比 - 绝对轨迹误差')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/comparison.png')
```

---

## 可视化

### 1. 使用 Rerun 可视化

```python
import base.rerun_ext as rre
from base.datatype import UnitData

# 初始化 Rerun
rre.rerun_init("session_name", imu_view_tags=["GT", "Raw"])

# 发送 IMU 数据
data = UnitData("/path/to/unit")
rre.send_imu_data(data.imu_data, tag="Raw")

# 发送真值位姿
rre.send_pose_data(data.gt_data, "GT")

# 发送预测位姿
from scipy.spatial.transform import Rotation
import pandas as pd

df = pd.read_csv(f"results/{data.name}/model1.csv")
t_us = df['#timestamp [us]'].values
ps = df[['p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]']].values
qs = df[['q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []']].values

pred_poses = PosesData(
    t_us=t_us,
    rots=Rotation.from_quat(qs.values, scalar_first=True),
    ps=ps
)
rre.send_pose_data(pred_poses, "model1")
```

### 2. 使用 DrawCompare 脚本

```bash
# 基础用法
python DrawCompare.py -u /path/to/unit -m model1.pt model2.pt

# 选项
--using_fusion    # 使用融合算法
--time_range 0 100 # 指定时间范围
```

### 3. 自定义可视化

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 读取数据
gt_df = pd.read_csv("/path/to/gt.csv")
pred_df = pd.read_csv("results/unit_name/model1.csv")

# 提取轨迹
gt_traj = gt_df[['tx', 'ty', 'tz']].values
pred_traj = pred_df[['p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]']].values

# 绘制3D轨迹
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], 
        label='真值', linewidth=2, color='blue')
ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], 
        label='预测', linewidth=2, color='red', linestyle='--')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend()
ax.set_title('轨迹对比')
plt.savefig('trajectory_comparison.png')
```

---

## 高级用法

### 1. 自定义数据处理

```python
from base.datatype import ImuData, UnitData
from base.interpolate import interpolate_vector3, slerp_rotation
from scipy.spatial.transform import Rotation

# 加载原始数据
data = UnitData("/path/to/unit")

# 自定义滤波
def low_pass_filter(data, alpha=0.95):
    filtered = np.zeros_like(data)
    filtered[0] = data[0]
    for i in range(1, len(data)):
        filtered[i] = alpha * filtered[i-1] + (1 - alpha) * data[i]
    return filtered

# 应用滤波
data.imu_data.acce = low_pass_filter(data.imu_data.acce, alpha=0.95)
data.imu_data.gyro = low_pass_filter(data.imu_data.gyro, alpha=0.95)

# 继续处理...
```

### 2. 批量处理脚本

```python
#!/usr/bin/env python3
"""批量处理数据集"""

from base.datatype import DeviceDataset
from base.model import ModelLoader, DataRunner, InertialNetworkData
import json
from pathlib import Path

def process_dataset(dataset_path, models_dir, output_dir):
    # 加载数据集和模型
    dataset = DeviceDataset(dataset_path)
    loader = ModelLoader(models_dir)
    models = loader.get_networks()
    
    results = {}
    
    for unit_data in dataset:
        print(f"处理: {unit_data.name}")
        
        try:
            # 运行预测
            runner = DataRunner(unit_data, InertialNetworkData.set_step(20))
            runner.predict_batch(models)
            
            # 记录结果
            results[unit_data.name] = {
                "status": "success",
                "imu_length": len(unit_data.imu_data),
                "gt_length": len(unit_data.gt_data)
            }
            
        except Exception as e:
            print(f"  错误: {e}")
            results[unit_data.name] = {
                "status": "failed",
                "error": str(e)
            }
    
    # 保存摘要
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    process_dataset(
        dataset_path="/path/to/dataset",
        models_dir="models",
        output_dir="results"
    )
```

### 3. 集成到训练流程

```python
#!/usr/bin/env python3
"""训练过程中的验证脚本"""

import torch
from base.model import ModelLoader, DataRunner, InertialNetworkData
from base.datatype import UnitData
from base.evaluate import Evaluation

def validate_model(model_path, validation_data_path):
    """验证模型性能"""
    
    # 加载模型和数据
    loader = ModelLoader("models")
    model = loader.get_by_name(model_path)
    data = UnitData(validation_data_path)
    
    # 运行预测
    runner = DataRunner(data, InertialNetworkData.set_step(20))
    result = runner.predict(model)
    
    # 评估
    evaluator = Evaluation(data.gt_data)
    
    # 读取预测结果
    import pandas as pd
    from scipy.spatial.transform import Rotation
    from base.datatype import PosesData
    
    df = pd.read_csv(f"results/{data.name}/{model.name}.csv")
    pred_poses = PosesData(
        t_us=df['#timestamp [us]'].values,
        rots=Rotation.from_quat(df[['q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []']].values, scalar_first=True),
        ps=df[['p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]']].values
    )
    
    evaluator.get_eval(pred_poses, model.name)
    
    # 返回关键指标
    return evaluator.resdict

# 示例：在训练循环中使用
metrics = validate_model("model_epoch_10.pt", "/path/to/validation/unit")
print(f"验证 ATE: {metrics['model_epoch_10.pt_ate_vec'][0]:.4f} m")
```

### 4. 参数调优

```python
#!/usr/bin/env python3
"""参数网格搜索"""

from base.model import DataRunner, InertialNetworkData, ModelLoader
from base.datatype import UnitData
from base.evaluate import Evaluation

def grid_search(data_path, model_name, steps=[10, 20, 40], rates=[100, 200]):
    """搜索最优的 step 和 rate 参数"""
    
    loader = ModelLoader("models")
    model = loader.get_by_name(model_name)
    data = UnitData(data_path)
    
    best_score = float('inf')
    best_params = None
    
    results = []
    
    for step in steps:
        for rate in rates:
            print(f"测试 step={step}, rate={rate}")
            
            try:
                # 运行预测
                DataClass = InertialNetworkData.set_step(step).set_rate(rate)
                runner = DataRunner(data, DataClass)
                result = runner.predict(model)
                
                # 评估
                evaluator = Evaluation(data.gt_data)
                # ... 读取和评估预测结果 ...
                
                # 获取 ATE
                ate = evaluator.resdict[f"{model.name}_ate_vec"][0]
                
                results.append({
                    'step': step,
                    'rate': rate,
                    'ate': ate
                })
                
                if ate < best_score:
                    best_score = ate
                    best_params = {'step': step, 'rate': rate}
                    
            except Exception as e:
                print(f"  失败: {e}")
    
    print(f"\n最优参数: {best_params}")
    print(f"最优 ATE: {best_score:.4f} m")
    
    return results, best_params

# 运行搜索
results, best = grid_search(
    data_path="/path/to/unit",
    model_name="model1.pt"
)
```

---

## 常见问题

### Q1: 如何处理不同采样率的数据？

```python
from base.interpolate import get_time_series
from base.datatype import ImuData

# 假设 IMU 是 200Hz，GT 是 10Hz
imu_data = ImuData.from_csv("imu.csv")
gt_data = GroundTruthData.from_csv("gt.csv")

# 统一到 200Hz
t_new_us = get_time_series([imu_data.t_us, gt_data.t_us], rate=200)

# 插值
imu_aligned = imu_data.interpolate(t_new_us)
gt_aligned = gt_data.interpolate(t_new_us)
```

### Q2: 如何处理数据缺失？

```python
import numpy as np
from scipy.interpolate import interp1d

# 填充缺失的 IMU 数据
def fill_missing_data(data, t_us):
    """使用线性插值填充缺失数据"""
    
    # 创建掩码
    valid = ~np.isnan(data).any(axis=1)
    
    if np.all(valid):
        return data
    
    # 对每个维度进行插值
    filled = np.zeros_like(data)
    for dim in range(data.shape[1]):
        valid_t = t_us[valid]
        valid_data = data[valid, dim]
        
        f = interp1d(valid_t, valid_data, kind='linear', 
                     bounds_error=False, fill_value='extrapolate')
        filled[:, dim] = f(t_us)
    
    return filled

# 使用
imu_data = ImuData.from_csv("imu.csv")
imu_data.gyro = fill_missing_data(imu_data.gyro, imu_data.t_us)
imu_data.acce = fill_missing_data(imu_data.acce, imu_data.t_us)
```

### Q3: 如何加速推理？

```python
import torch

# 1. 使用 GPU
from base import device
device.CpuDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 增加 batch size
DataClass = InertialNetworkData.set_step(40)  # 更大的步长

# 3. 预加载模型
loader = ModelLoader("models")
models = loader.get_networks()  # 一次性加载所有模型

# 4. 禁用可视化（如果不需要）
runner = DataRunner(data, DataClass, has_init_rerun=False)
```

### Q4: 如何调试预测错误？

```python
# 1. 检查输入数据
runner = DataRunner(data, InertialNetworkData.set_step(20))

# 查看数据块
for idx, block in runner.in_data.get_block_idx():
    print(f"Block {idx}: shape={block.shape}, mean={block.mean()}, std={block.std()}")
    break  # 只看第一个块

# 2. 检查模型输出
model = loader.get_by_name("model1.pt")
block = next(runner.in_data.get_block())
output = model.predict(block)
print(f"Output shape: {output[0].shape}")
print(f"Covariance shape: {output[1].shape}")

# 3. 可视化中间结果
import matplotlib.pyplot as plt
plt.figure()
plt.plot(block[0, 0, :])  # 陀螺仪 x
plt.plot(block[0, 3, :])  # 加速度计 x
plt.legend(['Gyro X', 'Acce X'])
plt.savefig('debug_input.png')
```

### Q5: 如何导出结果为其他格式？

```python
# 导出为 ROS bag 格式
import rosbag
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped

def export_to_rosbag(csv_path, bag_path, topic="/pose"):
    """导出为 ROS bag"""
    
    bag = rosbag.Bag(bag_path, 'w')
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        msg = PoseStamped()
        
        # Header
        msg.header.stamp = rospy.Time.from_sec(row['#timestamp [us]'] / 1e6)
        msg.header.frame_id = "world"
        
        # Position
        msg.pose.position.x = row['p_RS_R_x [m]']
        msg.pose.position.y = row['p_RS_R_y [m]']
        msg.pose.position.z = row['p_RS_R_z [m]']
        
        # Orientation
        msg.pose.orientation.w = row['q_RS_w []']
        msg.pose.orientation.x = row['q_RS_x []']
        msg.pose.orientation.y = row['q_RS_y []']
        msg.pose.orientation.z = row['q_RS_z []']
        
        bag.write(topic, msg)
    
    bag.close()
    print(f"导出到: {bag_path}")

# 使用
export_to_rosbag("results/unit_name/model1.csv", "trajectory.bag")
```

### Q6: 内存不足怎么办？

```python
# 1. 分批处理
def process_in_batches(data_path, model_name, batch_size=100):
    """分批处理长序列数据"""
    
    loader = ModelLoader("models")
    model = loader.get_by_name(model_name)
    data = UnitData(data_path)
    
    # 分割数据
    total_length = len(data.gt_data)
    for start in range(0, total_length, batch_size):
        end = min(start + batch_size, total_length)
        
        print(f"处理批次: {start}-{end}")
        
        # 创建临时数据
        # ... 处理当前批次 ...
        
        # 清理内存
        import gc
        gc.collect()

# 2. 使用更小的 step
DataClass = InertialNetworkData.set_step(40)  # 减少预测频率
```

---

## 更多资源

- [项目主页](README.md)
- [API 文档](#) (待完善)
- [论文引用](#) (待完善)
- [贡献指南](README.md#贡献)

## 许可证

请参考项目根目录的 LICENSE 文件。