# ModelVerify 快速参考

> **文档更新日期：2026-01-08**

一份简洁的快速参考指南，帮助你快速开始使用 ModelVerify。

## 快速安装

```bash
# 克隆并安装
git clone <repo-url>
cd ModelVerify
uv sync
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows
```

## 命令行速查

### 基本命令

```bash
# 验证单个数据单元
python main.py -u /path/to/unit -m model1.pt

# 验证整个数据集
python main.py -d /path/to/dataset -m model1.pt

# 多模型对比
python main.py -u /path/to/unit -m model1.pt model2.pt

# 指定时间范围（秒）
python main.py -u /path/to/unit -m model1.pt --time_range 0 100

# 使用AHRS数据
python main.py -u /path/to/unit -m model1.pt --using_ahrs
```

### 可视化脚本

```bash
# 对比可视化
python DrawCompare.py -u /path/to/unit -m model1.pt model2.pt

# 查看TLIO数据
python TLIOView.py -u /path/to/unit

# 旋转对齐可视化（新生成4张图）
python visualize_rotation.py -u /path/to/unit
# 批量处理
python visualize_rotation.py -d /path/to/dataset
```

**旋转对齐可视化生成的图表**：
- `rotation_comparison.png` - 不同校准方式下的角速度对比
- `time_alignment.png` - 时间对齐前后效果对比
- `gt_trajectory.png` - Ground Truth 轨迹（3D位置、姿态、速度）
- `calibration_summary.png` - 校准效果汇总柱状图

## Python API 速查

### 加载数据

```python
from base.datatype import UnitData, DeviceDataset

# 加载单个单元
data = UnitData("/path/to/unit")
print(f"名称: {data.name}")
print(f"IMU数据: {len(data.imu_data)} 点")
print(f"GT数据: {len(data.gt_data)} 点")

# 加载数据集
dataset = DeviceDataset("/path/to/dataset")
print(f"共 {len(dataset)} 个单元")
```

### 加载模型

```python
from base.model import ModelLoader

# 创建加载器
loader = ModelLoader("models")

# 列出所有模型
for model_path in loader.models:
    print(model_path.name)

# 加载单个模型
model = loader.get_by_name("model1.pt")

# 加载多个模型
models = loader.get_by_names(["model1.pt", "model2.pt"])
```

### 运行预测

```python
from base.model import DataRunner, InertialNetworkData

# 单模型预测
runner = DataRunner(data, InertialNetworkData.set_step(20))
result = runner.predict(model)

# 批量预测
runner = DataRunner(data, InertialNetworkData.set_step(20))
results = runner.predict_batch(models)
```

### 结果评估

```python
from base.evaluate import Evaluation
from base.datatype import PosesData
import pandas as pd
from scipy.spatial.transform import Rotation

# 读取预测结果
df = pd.read_csv("results/unit_name/model1.csv")
pred_poses = PosesData(
    t_us=df['#timestamp [us]'].values,
    rots=Rotation.from_quat(
        df[['q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []']].values,
        scalar_first=True
    ),
    ps=df[['p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]']].values
)

# 评估
evaluator = Evaluation(data.gt_data)
evaluator.get_eval(pred_poses, "model1")
evaluator.print()
```

### 数据操作

```python
from scipy.spatial.transform import Rotation
import numpy as np

# 时间范围
imu_range = data.imu_data.get_time_range((10.0, 20.0))

# 插值
new_times = np.linspace(0, 1e6, 100)
imu_interp = data.imu_data.interpolate(new_times)

# 坐标变换
global_imu = data.imu_data.transform()

# 位姿操作
pose = Pose(Rotation.identity(), np.array([1, 0, 0]))
pose2 = pose.compose(Pose.identity())
inv_pose = pose.inverse()
```

## 数据格式速查

### IMU 数据 (imu.csv)

```csv
timestamp,gyro_x,gyro_y,gyro_z,acce_x,acce_y,acce_z,qw,qx,qy,qz,time_offset[,magn_x,magn_y,magn_z]
```

**单位**: 时间戳(μs), 陀螺仪(rad/s), 加速度计(m/s²), 四元数

### GT 数据 (gt.csv)

```csv
timestamp,tx,ty,tz,qw,qx,qy,qz
```

**单位**: 时间戳(μs), 位置(m), 四元数

### 结果 CSV

```csv
#timestamp [us],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],cov_x [],cov_y [],cov_z []
```

## 配置参数速查

### InertialNetworkData 参数

```python
DataClass = InertialNetworkData
    .set_step(20)      # 预测步长（IMU数据点数）
    .set_rate(200)     # IMU采样率（Hz）
    .remove_gravity()  # 移除重力加速度
    .set_time_range((0, 100))  # 时间范围（秒）
```

### DataRunner 参数

```python
runner = DataRunner(
    data,                      # UnitData 对象
    DataClass,                 # 数据处理类
    time_range=(0, 100),       # 时间范围（秒）
    has_init_rerun=False,      # 是否初始化Rerun
    using_gt=True              # 是否使用GT数据
)
```

## 评估指标说明

| 指标 | 全称 | 说明 | 单位 |
|------|------|------|------|
| **ATE** | Absolute Trajectory Error | 绝对轨迹误差 | 米 |
| **APE** | Absolute Pose Error | 绝对位姿误差 | 米/弧度 |
| **RTE** | Relative Pose Error | 相对位姿误差（默认60秒间隔） | 米/弧度 |

## 常见问题速查

**Q: 如何解决 OOM？**
```python
# 减小 batch size 或 step
DataClass = InertialNetworkData.set_step(40)
```

**Q: 如何加速推理？**
```python
# 使用 GPU
import torch
from base import device
device.CpuDevice = torch.device("cuda")
```

**Q: 数据不匹配？**
```python
# 检查时间对齐
print(f"IMU: {data.imu_data.t_us[0]} - {data.imu_data.t_us[-1]}")
print(f"GT:  {data.gt_data.t_us[0]} - {data.gt_data.t_us[-1]}")
```

**Q: 如何禁用可视化？**
```python
runner = DataRunner(data, DataClass, has_init_rerun=False)
```

## 目录结构速查

```
ModelVerify/
├── main.py              # 主程序
├── DrawCompare.py       # 对比可视化
├── base/
│   ├── datatype.py      # 数据类型
│   ├── model.py         # 模型加载
│   ├── evaluate.py      # 评估
│   └── interpolate.py   # 插值
├── models/              # 模型文件
├── datasets/            # 数据集
└── results/             # 结果输出
```

## 更多资源

- 📖 [完整使用指南](USAGE.md) - 详细文档和教程
- 🏠 [项目主页](../README.md) - 项目概述和功能介绍
- 💬 [GitHub Issues](../../issues) - 问题反馈和讨论

## 快速链接

- [环境配置](USAGE.md#环境配置)
- [数据准备](USAGE.md#数据准备)
- [Python API](USAGE.md#python-api-使用)
- [结果评估](USAGE.md#结果评估)
- [常见问题](USAGE.md#常见问题)