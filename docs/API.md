# ModelVerify API 参考文档

> **文档更新日期：2026-01-08**

本文档提供了 ModelVerify 项目中所有主要 API 的详细参考。

## 目录

- [数据类型模块 (`base.datatype`)](#数据类型模块-basedatatype)
- [模型模块 (`base.model`)](#模型模块-basemodel)
- [评估模块 (`base.evaluate`)](#评估模块-baseevaluate)
- [插值模块 (`base.interpolate`)](#插值模块-baseinterpolate)
- [工具模块 (`base.rerun_ext`)](#工具模块-basererun_ext)

---

## 数据类型模块 (`base.datatype`)

### Pose

表示刚体位姿（旋转 + 平移）。

```python
from base.datatype import Pose
from scipy.spatial.transform import Rotation
import numpy as np

# 创建位姿
rot = Rotation.from_euler('xyz', [0, 0, 90], degrees=True)
trans = np.array([1.0, 2.0, 3.0])
pose = Pose(rot, trans, t_us=1000)
```

#### 类方法

##### `Pose.identity()`

创建单位位姿（零旋转、零平移）。

**返回:** `Pose` - 单位位姿

```python
identity_pose = Pose.identity()
```

##### `Pose.from_rotation(rot: Rotation)`

从旋转创建位姿（平移为零）。

**参数:**
- `rot` (Rotation): 旋转对象

**返回:** `Pose` - 位姿对象

```python
pose = Pose.from_rotation(Rotation.from_euler('z', 45, degrees=True))
```

##### `Pose.from_transform(trans: NDArray)`

从平移向量创建位姿（旋转为单位矩阵）。

**参数:**
- `trans` (NDArray): 3维平移向量

**返回:** `Pose` - 位姿对象

```python
pose = Pose.from_transform(np.array([1, 0, 0]))
```

#### 实例方法

##### `copy()`

创建位姿的深拷贝。

**返回:** `Pose` - 位姿副本

##### `compose(other: Pose) -> Pose`

位姿组合：`self @ other`

**参数:**
- `other` (Pose): 另一个位姿

**返回:** `Pose` - 组合后的位姿

```python
result = pose1.compose(pose2)
```

##### `compose_self(other: Pose)`

就地组合位姿。

**参数:**
- `other` (Pose): 另一个位姿

##### `compose_trans_self(trans: NDArray)`

应用平移变换。

**参数:**
- `trans` (NDArray): 3维平移向量

##### `inverse() -> Pose`

求位姿的逆。

**返回:** `Pose` - 逆位姿

```python
inv_pose = pose.inverse()
```

##### `get_yaw_pose() -> Pose`

提取偏航角位姿（只保留Z轴旋转）。

**返回:** `Pose` - 仅含偏航的位姿

---

### PosesData

位姿序列容器，用于存储和处理多个位姿。

```python
from base.datatype import PosesData
from scipy.spatial.transform import Rotation
import numpy as np

t_us = np.array([0, 1e6, 2e6])
rots = Rotation.from_euler('xyz', [[0,0,0], [1,0,0], [2,0,0]], degrees=True)
ps = np.array([[0,0,0], [1,0,0], [2,0,0]])

poses = PosesData(t_us, rots, ps)
```

#### 类方法

##### `PosesData.from_list(pose_list: list[Pose])`

从位姿列表创建。

**参数:**
- `pose_list` (list[Pose]): 位姿列表

**返回:** `PosesData` - 位姿序列对象

#### 实例属性

##### `rate: float`

位姿序列的平均采样率（Hz）。

```python
print(f"采样率: {poses.rate:.2f} Hz")
```

##### `length_meter: float`

轨迹总长度（米）。

```python
print(f"轨迹长度: {poses.length_meter:.2f} m")
```

#### 实例方法

##### `get_pose(index: int = 0) -> Pose`

获取指定索引的位姿。

**参数:**
- `index` (int): 位姿索引

**返回:** `Pose` - 位姿对象

##### `get_time_range(time_range: tuple[float | None, float | None]) -> PosesData`

获取时间范围内的位姿。

**参数:**
- `time_range` (tuple): (起始秒, 结束秒)，None表示不限制

**返回:** `PosesData` - 筛选后的位姿序列

```python
# 获取10-20秒的位姿
subset = poses.get_time_range((10.0, 20.0))
```

##### `interpolate(t_new_us: NDArray, bounds_error: bool = False) -> PosesData`

插值到新的时间序列。

**参数:**
- `t_new_us` (NDArray): 新的时间戳数组（微秒）
- `bounds_error` (bool): 是否在超出范围时报错

**返回:** `PosesData` - 插值后的位姿序列

```python
new_times = np.linspace(0, 5e6, 1000)
interpolated = poses.interpolate(new_times)
```

##### `transform_local(tf: Pose)`

应用局部坐标系变换。

**参数:**
- `tf` (Pose): 变换位姿

##### `transform_global(tf: Pose)`

应用全局坐标系变换。

**参数:**
- `tf` (Pose): 变换位姿

##### `length_meter_fix_rate(rate: int = 1) -> float`

按固定采样率计算轨迹长度。

**参数:**
- `rate` (int): 采样率（Hz）

**返回:** `float` - 轨迹长度（米）

---

### ImuData

IMU传感器数据容器。

```python
from base.datatype import ImuData
from pathlib import Path

# 从CSV加载
imu_data = ImuData.from_csv(Path("/path/to/imu.csv"))
```

#### 类方法

##### `ImuData.from_raw(raw: NDArray) -> ImuData`

从原始numpy数组创建。

**参数:**
- `raw` (NDArray): 原始数据数组，形状为(N, >=12)

**返回:** `ImuData` - IMU数据对象

##### `ImuData.from_csv(path: Path) -> ImuData`

从CSV文件加载。

**参数:**
- `path` (Path): CSV文件路径

**返回:** `ImuData` - IMU数据对象

#### 实例属性

- `t_us` (NDArray): 时间戳数组（微秒）
- `gyro` (NDArray): 陀螺仪数据 (N, 3)，单位 rad/s
- `acce` (NDArray): 加速度计数据 (N, 3)，单位 m/s²
- `ahrs` (Rotation): AHRS姿态估计
- `magn` (NDArray): 磁力计数据 (N, 3)
- `frame` (str): 坐标系类型，"local" 或 "global"

#### 实例方法

##### `interpolate(t_new_us: NDArray, bounds_error: bool = False) -> ImuData`

插值到新的时间序列。

**参数:**
- `t_new_us` (NDArray): 新的时间戳数组
- `bounds_error` (bool): 是否在超出范围时报错

**返回:** `ImuData` - 插值后的IMU数据

##### `transform(rots: Rotation | None = None) -> ImuData`

变换到全局坐标系。

**参数:**
- `rots` (Rotation): 使用的旋转，默认为AHRS姿态

**返回:** `ImuData` - 变换后的IMU数据

##### `get_time_range(time_range: tuple[float | None, float | None]) -> ImuData`

获取时间范围内的数据。

**参数:**
- `time_range` (tuple): (起始秒, 结束秒)

**返回:** `ImuData` - 筛选后的IMU数据

##### `to_poses() -> PosesData`

转换为位姿序列。

**返回:** `PosesData` - 位姿序列（位置为零）

---

### GroundTruthData

真值位姿数据，继承自 `PosesData`。

```python
from base.datatype import GroundTruthData
from pathlib import Path

gt_data = GroundTruthData.from_csv(Path("/path/to/gt.csv"))
```

#### 类方法

##### `GroundTruthData.from_raw(raw: NDArray) -> GroundTruthData`

从原始数组创建。

**参数:**
- `raw` (NDArray): 原始数据 (N, 8)，包含时间戳、位置、四元数

**返回:** `GroundTruthData` - 真值数据对象

##### `GroundTruthData.from_csv(path: Path) -> GroundTruthData`

从CSV文件加载。

**参数:**
- `path` (Path): CSV文件路径

**返回:** `GroundTruthData` - 真值数据对象

---

### UnitData

单个数据单元，整合了IMU、GT、标定等所有数据。

```python
from base.datatype import UnitData

unit_data = UnitData("/path/to/unit")
```

#### 初始化参数

##### `__init__(base_dir: Path | str, using_ext: int = True)`

**参数:**
- `base_dir` (Path | str): 数据单元目录路径
- `using_ext` (int): 是否加载外部标定文件

#### 实例属性

- `name` (str): 单元名称
- `device_name` (str): 设备名称
- `imu_data` (ImuData): IMU数据
- `gt_data` (GroundTruthData): 真值数据
- `fusion_data` (PosesData): 融合数据（可选）
- `calib_data` (CalibrationData): 标定数据
- `check_data` (DataCheck): 数据检查信息
- `has_fusion` (bool): 是否包含融合数据

---

### DeviceDataset

设备数据集，包含多个数据单元。

```python
from base.datatype import DeviceDataset

dataset = DeviceDataset("/path/to/dataset")
for unit_data in dataset:
    print(unit_data.name)
```

#### 初始化参数

##### `__init__(base_dir: Path | str, using_ext: bool = True)`

**参数:**
- `base_dir` (Path | str): 数据集目录路径
- `using_ext` (bool): 是否加载外部标定文件

#### 实例方法

##### `__getitem__(index: int) -> UnitData`

获取指定索引的数据单元。

##### `__len__() -> int`

返回数据单元数量。

---

## 模型模块 (`base.model`)

### InertialNetwork

基础神经网络类，用于加载TorchScript模型。

```python
from base.model import InertialNetwork

network = InertialNetwork("models/model.pt")
measurement, covariance = network.predict(input_block)
```

#### 初始化参数

##### `__init__(model_path: Path | str, input_shape: tuple[int, ...] | None = None)`

**参数:**
- `model_path` (Path | str): 模型文件路径
- `input_shape` (tuple | None): 输入形状，用于验证

#### 实例方法

##### `predict(block: NDArray) -> tuple[NDArray, NDArray]`

运行模型预测。

**参数:**
- `block` (NDArray): 输入数据块，形状为(1, 6, rate)

**返回:** 
- `measurement` (NDArray): 预测的速度增量
- `covariance` (NDArray): 协方差矩阵

##### `reset()`

重置模型到初始状态。

---

### MemoryInertialNetwork

支持LSTM记忆的神经网络类。

```python
from base.model import MemoryInertialNetwork

network = MemoryInertialNetwork("models/lstm_model.pt", polling_rate=10)
```

#### 初始化参数

##### `__init__(model_path: Path | str, input_shape: tuple[int, ...] | None = None, polling_rate=10)`

**参数:**
- `model_path` (Path | str): 模型文件路径
- `input_shape` (tuple | None): 输入形状
- `polling_rate` (int): 记忆轮询率

#### 实例方法

##### `predict(block: NDArray) -> tuple[NDArray, NDArray]`

运行模型预测，维护LSTM记忆状态。

**参数:**
- `block` (NDArray): 输入数据块

**返回:** (measurement, covariance)

##### `reset()`

重置模型和记忆状态。

---

### ModelLoader

模型加载器，支持批量加载模型。

```python
from base.model import ModelLoader

loader = ModelLoader("models")
models = loader.get_by_names(["model1.pt", "model2.pt"])
```

#### 初始化参数

##### `__init__(base_dir: Path | str)`

**参数:**
- `base_dir` (Path | str): 模型目录路径

#### 实例方法

##### `get_by_name(name: str) -> InertialNetwork | MemoryInertialNetwork`

根据名称加载单个模型。

**参数:**
- `name` (str): 模型名称（可包含部分匹配）

**返回:** InertialNetwork 或 MemoryInertialNetwork

##### `get_by_names(names: list[str]) -> list`

批量加载模型。

**参数:**
- `names` (list): 模型名称列表

**返回:** 模型列表

##### `get_networks() -> list[InertialNetwork]`

加载所有模型。

**返回:** 模型列表

---

### InertialNetworkData

IMU数据处理器，为神经网络准备输入数据。

```python
from base.model import InertialNetworkData, DataRunner
from base.datatype import UnitData

data = UnitData("/path/to/unit")
DataClass = InertialNetworkData.set_step(20).set_rate(200)
runner = DataRunner(data, DataClass)
```

#### 类方法

##### `set_step(step: int) -> type`

设置预测步长。

**参数:**
- `step` (int): 步长（IMU数据点数）

**返回:** 类本身（链式调用）

##### `set_rate(rate: int) -> type`

设置IMU采样率。

**参数:**
- `rate` (int): 采样率（Hz）

**返回:** 类本身

##### `remove_gravity() -> type`

启用重力移除。

**返回:** 类本身

##### `set_time_range(time_range: tuple[float, float]) -> type`

设置时间范围。

**参数:**
- `time_range` (tuple): (起始秒, 结束秒)

**返回:** 类本身

#### 实例方法

##### `predict_using(net: InertialNetwork, ref_poses: PosesData | None = None) -> NetworkResult`

使用单个模型预测。

**参数:**
- `net` (InertialNetwork): 神经网络模型
- `ref_poses` (PosesData | None): 参考位姿序列

**返回:** NetworkResult - 预测结果

##### `predict_usings(networks: list[InertialNetwork], ref_poses: PosesData) -> list[NetworkResult]`

批量预测多个模型。

**参数:**
- `networks` (list): 模型列表
- `ref_poses` (PosesData): 参考位姿序列

**返回:** NetworkResult列表

##### `predict_rot(network: InertialNetwork, ref_poses: PosesData, degrees: list = [0]) -> list[NetworkResult]`

使用不同旋转角度预测。

**参数:**
- `network` (InertialNetwork): 神经网络模型
- `ref_poses` (PosesData): 参考位姿序列
- `degrees` (list): 旋转角度列表（度）

**返回:** NetworkResult列表

---

### DataRunner

数据运行器，整合数据处理、模型推理和结果保存。

```python
from base.model import DataRunner, InertialNetworkData, ModelLoader
from base.datatype import UnitData

data = UnitData("/path/to/unit")
loader = ModelLoader("models")
model = loader.get_by_name("model1.pt")

runner = DataRunner(data, InertialNetworkData.set_step(20))
result = runner.predict(model)
```

#### 初始化参数

##### `__init__(ud: UnitData, Data: type[InertialNetworkData], *, time_range: tuple[float | None, float | None] = (None, None), has_init_rerun: bool = False, using_gt: bool = True)`

**参数:**
- `ud` (UnitData): 数据单元
- `Data` (type): 数据处理类
- `time_range` (tuple): 时间范围（秒）
- `has_init_rerun` (bool): 是否初始化Rerun可视化
- `using_gt` (bool): 是否使用GT数据

#### 实例方法

##### `predict(net: InertialNetwork) -> NetworkResult`

运行单个模型预测并保存结果。

**参数:**
- `net` (InertialNetwork): 神经网络模型

**返回:** NetworkResult - 预测结果

##### `predict_batch(networks: list[InertialNetwork]) -> list[NetworkResult]`

批量预测并保存结果。

**参数:**
- `networks` (list): 模型列表

**返回:** NetworkResult列表

##### `predict_rot(network: InertialNetwork, degrees: list = [0]) -> list[NetworkResult]`

旋转增强预测。

**参数:**
- `network` (InertialNetwork): 神经网络模型
- `degrees` (list): 旋转角度列表

**返回:** NetworkResult列表

---

### NetworkResult

神经网络预测结果容器。

#### 实例方法

##### `add(output: tuple[NDArray, NDArray], ref_pose: Pose = Pose.identity()) -> Pose`

添加预测结果。

**参数:**
- `output` (tuple): (measurement, covariance)
- `ref_pose` (Pose): 参考位姿

**返回:** Pose - 当前位姿

##### `to_csv(path: Path | str)`

保存结果为CSV文件。

**参数:**
- `path` (Path | str): 输出文件路径

---

## 评估模块 (`base.evaluate`)

### Evaluation

性能评估类，计算ATE、APE、RPE等指标。

```python
from base.evaluate import Evaluation
from base.datatype import PosesData

evaluator = Evaluation(gt_poses)
results = evaluator.get_eval(pred_poses, "model1")
evaluator.print()
```

#### 初始化参数

##### `__init__(ref_poses: PosesData)`

**参数:**
- `ref_poses` (PosesData): 参考位姿序列（真值）

#### 实例属性

- `length` (float): 参考轨迹长度
- `time_length` (float): 参考轨迹时间长度（秒）
- `mean_velocity` (float): 平均速度（米/秒）
- `rte_duration` (int): RTE计算的时间间隔（秒），默认60

#### 实例方法

##### `get_eval(eva_poses: PosesData, tag: str) -> dict`

评估预测位姿。

**参数:**
- `eva_poses` (PosesData): 预测位姿序列
- `tag` (str): 结果标签

**返回:** dict - 评估结果字典，包含：
- `{tag}_ape_vec`: APE向量
- `{tag}_ate_vec`: ATE向量
- `{tag}_rpe_vec`: RPE向量

##### `print()`

打印评估结果。

##### `save(file: Path | str)`

保存评估结果为JSON文件。

**参数:**
- `file` (Path | str): 输出文件路径

---

## 插值模块 (`base.interpolate`)

### 时间序列插值

```python
from base.interpolate import get_time_series

# 统一时间序列
t_new = get_time_series([imu_data.t_us, gt_data.t_us], rate=200)
```

#### 函数

##### `get_time_series(t_lists: list[NDArray], rate: int | None = None) -> NDArray`

生成统一的时间序列。

**参数:**
- `t_lists` (list[NDArray]): 多个时间戳列表
- `rate` (int | None): 采样率（Hz），None则使用最小时间步长

**返回:** NDArray - 统一的时间戳数组

##### `interpolate_vector3(data: NDArray, t_orig: NDArray, t_new: NDArray, bounds_error: bool = False) -> NDArray`

三维向量插值。

**参数:**
- `data` (NDArray): 原始数据 (N, 3)
- `t_orig` (NDArray): 原始时间戳
- `t_new` (NDArray): 新时间戳
- `bounds_error` (bool): 是否在超出范围时报错

**返回:** NDArray - 插值后的数据 (M, 3)

##### `slerp_rotation(rots: Rotation, t_orig: NDArray, t_new: NDArray) -> Rotation`

旋转插值（SLERP）。

**参数:**
- `rots` (Rotation): 原始旋转序列
- `t_orig` (NDArray): 原始时间戳
- `t_new` (NDArray): 新时间戳

**返回:** Rotation - 插值后的旋转序列

---

## 工具模块 (`base.rerun_ext`)

### Rerun可视化扩展

```python
import base.rerun_ext as rre
from base.datatype import ImuData, PosesData

# 初始化
rre.rerun_init("session", imu_view_tags=["GT", "Raw"])

# 发送IMU数据
rre.send_imu_data(imu_data, tag="GT")

# 发送位姿数据
rre.send_pose_data(poses, "prediction")
```

#### 函数

##### `rerun_init(session_name: str, imu_view_tags: list[str] | None = None)`

初始化Rerun会话。

**参数:**
- `session_name` (str): 会话名称
- `imu_view_tags` (list[str] | None): IMU视图标签列表

##### `send_imu_data(imu_data: ImuData, tag: str)`

发送IMU数据到Rerun。

**参数:**
- `imu_data` (ImuData): IMU数据
- `tag` (str): 数据标签

##### `send_pose_data(poses: PosesData, tag: str, color: list[int] | None = None)`

发送位姿数据到Rerun。

**参数:**
- `poses` (PosesData): 位姿序列
- `tag` (str): 数据标签
- `color` (list[int]): RGB颜色，如[255, 0, 0]

##### `log_network_pose(t_us: int, pose: Pose, path: list[NDArray], tag: str)`

记录网络预测位姿。

**参数:**
- `t_us` (int): 时间戳
- `pose` (Pose): 当前位姿
- `path` (list[NDArray]): 轨迹点列表
- `tag` (str): 数据标签

##### `log_coordinate(name: str, length: float = 1.0, labels: list[str] | None = None, show_labels: bool = True)`

记录坐标系。

**参数:**
- `name` (str): 坐标系名称
- `length` (float): 坐标轴长度
- `labels` (list[str] | None): 轴标签
- `show_labels` (bool): 是否显示标签

---

## 类型别名

### NetworkOutput

```python
NetworkOutput = tuple[NDArray, NDArray]
```

神经网络输出类型：(measurement, covariance)

### Frame

```python
Frame: TypeAlias = Literal["local", "global"]
```

坐标系类型："local" 或 "global"

---

## 常量

### 默认参数

```python
# InertialNetworkData 默认值
InertialNetworkData.step = 80      # 默认步长
InertialNetworkData.rate = 200     # 默认采样率

# Evaluation 默认值
Evaluation.rte_duration = 60       # RTE计算间隔（秒）
```

---

## 使用示例

### 完整工作流程

```python
from base.datatype import UnitData, PosesData
from base.model import ModelLoader, DataRunner, InertialNetworkData
from base.evaluate import Evaluation
import pandas as pd
from scipy.spatial.transform import Rotation

# 1. 加载数据
data = UnitData("/path/to/unit")

# 2. 加载模型
loader = ModelLoader("models")
model = loader.get_by_name("model1.pt")

# 3. 运行预测
runner = DataRunner(data, InertialNetworkData.set_step(20))
result = runner.predict(model)

# 4. 读取预测结果
df = pd.read_csv(f"results/{data.name}/{model.name}.csv")

# 5. 创建位姿序列
pred_poses = PosesData(
    t_us=df['#timestamp [us]'].values,
    rots=Rotation.from_quat(
        df[['q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []']].values,
        scalar_first=True
    ),
    ps=df[['p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]']].values
)

# 6. 评估性能
evaluator = Evaluation(data.gt_data)
evaluator.get_eval(pred_poses, model.name)
evaluator.print()

# 7. 保存评估结果
evaluator.save(f"results/{data.name}/{model.name}_eval.json")
```

---

## 注意事项

1. **时间单位**: 所有时间戳单位为微秒（μs）
2. **四元数格式**: 四元数使用标量在前格式（w, x, y, z）
3. **坐标系**: 
   - IMU数据默认在局部坐标系
   - 使用 `transform()` 方法变换到全局坐标系
4. **内存管理**: 处理大数据集时建议分批处理
5. **可视化**: 使用 `has_init_rerun=False` 禁用可视化以提高性能

---

## 更多资源

- [使用指南](USAGE.md) - 详细的使用教程
- [快速参考](QUICKSTART.md) - 常用命令速查
- [项目主页](../README.md) - 项目概述