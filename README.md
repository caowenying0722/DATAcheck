# ModelVerify

> **文档更新日期：2026-01-14**

一个用于验证和测试惯性导航模型的Python工具包。

## 功能特性

- 🧠 **模型加载与预测**: 支持加载PyTorch模型(.pt格式)进行惯性导航预测
- 📊 **数据处理**: 提供IMU数据和位姿数据的处理与插值功能
- 🎯 **批量验证**: 支持单个数据单元和整个数据集的批量模型验证
- 📈 **可视化**: 集成Rerun SDK进行数据可视化，支持CDF、不确定性、轨迹等图表
- 🔧 **灵活配置**: 支持命令行参数配置，可自定义模型和数据路径
- 🔄 **数据转换**: 支持从H5文件生成验证数据集
- 📱 **模型转换**: 支持TorchScript模型转换和Android部署
- 🎨 **高级可视化**: 支持CDF、不确定性分析、轨迹对比等可视化
- 📈 **性能评估**: 提供ATE/APE/RPE等多种评估指标
- 🧪 **测试覆盖**: 提供完整的单元测试和集成测试
- 📚 **完整文档**: 提供详细的API文档和使用示例

## 📚 文档

查看完整文档以获取更多信息：

- 📋 [文档中心](docs/README.md) - 文档导航和学习路径
- 🚀 [快速参考](docs/QUICKSTART.md) - 常用命令和API速查手册
- 📖 [详细使用指南](docs/USAGE.md) - 完整的数据格式说明和API使用教程
- 📚 [API参考文档](docs/API.md) - 所有类和函数的详细API文档

### 核心教程

- 🔧 [数据准备](docs/USAGE.md#数据准备) - 数据格式和标定说明
- 🧠 [模型推理](docs/USAGE.md#模型推理) - 模型加载和预测流程
- 📊 [结果评估](docs/USAGE.md#结果评估) - ATE/APE/RPE等评估指标
- 🎨 [可视化](docs/USAGE.md#可视化) - 使用Rerun进行数据可视化

### 高级主题

- 🚀 [高级用法](docs/USAGE.md#高级用法) - 最佳实践和自定义方法
- ❓ [常见问题](docs/USAGE.md#常见问题) - 问题排查和解决方案


## 安装

### 环境要求

- Python >= 3.11
- 推荐使用 `uv` 作为包管理器

### 安装依赖

```bash
# 使用uv安装依赖
uv sync

# 或使用pip安装
pip install -e .
```

## 快速开始

### 基本用法

```bash
# 验证单个数据单元
python main.py -u <unit_path> -m model1.pt model2.pt

# 验证整个数据集
python main.py -d <dataset_path> -m model1.pt model2.pt

# 使用AHRS数据
python main.py -u <unit_path> -m model1.pt --using_ahrs
```

### 模型验证

```bash
# 批量验证模型并生成CDF图
python VaildModel.py -d <dataset_path> -m model1.pt

# 模型分析
python ModelAnalysis.py -u <unit_path> -m model1.pt
```

### 数据集生成

```bash
# 从H5文件生成验证数据集
python GenerateFromH5.py -h <h5_file_path>
```

### 可视化

```bash
# 绘制模型轨迹
python DrawModel.py -u <unit_path> -m model1.pt

# 对比多个模型
python DrawCompare.py -u <unit_path> -m model1.pt model2.pt
```

### 📚 更多文档

查看 [详细使用指南](docs/USAGE.md) 获取更多信息：

- 📖 完整的数据格式说明
- 🔧 详细的API使用示例
- 📊 结果评估与可视化教程
- 🚀 高级用法和最佳实践
- ❓ 常见问题解答

### 参数说明

- `-u, --unit`: 指定单个数据单元路径
- `-d, --dataset`: 指定数据集路径
- `-m, --models`: 指定要使用的模型文件名(必需参数，可指定多个)
- `--models_path`: 指定模型文件夹路径(默认为"models")
- `--using_ahrs`: 使用AHRS数据而非GT数据旋转IMU数据
- `--time_range`: 指定时间范围进行验证
- `-h, --h5`: 指定H5文件路径(GenerateFromH5专用)

## 项目结构

```
ModelVerify/
├── main.py                 # 主程序入口
├── TLIOView.py             # TLIO数据可视化
├── VaildModel.py           # 模型验证工具
├── ModelAnalysis.py        # 模型分析工具
├── GenerateFromH5.py       # 从H5文件生成验证数据集
├── TorchScript.py          # TorchScript转换示例
├── TorchScript2Android.py  # TorchScript转换为Android
├── DrawModel.py            # 模型绘制工具
├── DrawCompare.py          # 模型对比可视化
├── DrawCompareOnly.py      # 仅对比可视化
├── SpinCompare.py          # 旋转对比分析
├── base/                   # 核心模块
│   ├── args_parser.py      # 命令行参数解析
│   ├── datatype.py         # 数据类型定义
│   ├── device.py           # 设备配置
│   ├── interpolate.py      # 数据插值
│   ├── model.py            # 模型加载与预测
│   ├── evaluate.py         # 性能评估
│   ├── rerun_ext.py        # Rerun扩展
│   ├── serialize.py        # 数据序列化
│   ├── rtab.py             # RTAB相关功能
│   ├── binary.py           # 二进制数据处理
│   ├── calibration/        # 标定模块
│   └── draw/               # 可视化模块
│       ├── CDF.py          # 累积分布函数绘制
│       ├── Uncertainty.py  # 不确定性可视化
│       └── Poses.py        # 位姿可视化
├── docs/                   # 详细文档
│   └── USAGE.md            # 使用指南
├── tests/                  # 测试用例
├── datasets/               # 数据集目录
└── results/                # 结果输出目录
```

## 核心组件

### InertialNetwork
负责加载和运行PyTorch模型的核心类：

```python
from base.model import InertialNetwork

# 加载模型
network = InertialNetwork("path/to/model.pt")

# 进行预测
measurement, covariance = network.predict(input_data)
```

### ModelLoader
模型加载器，支持批量加载和管理多个模型：

```python
from base.model import ModelLoader

loader = ModelLoader("models")
models = loader.get_by_names(["model1.pt", "model2.pt"])
```

### 数据类型
项目定义了以下核心数据类型：

- `Pose`: 位姿数据(旋转+平移)
- `ImuData`: IMU传感器数据
- `UnitData`: 单元数据容器
- `DeviceDataset`: 设备数据集
- `GroundTruthData`: 真值数据
- `PosesData`: 位姿序列数据

### 数据处理
提供完整的数据处理流程：

- 时间序列插值
- 旋转插值(SLERP)
- 向量插值
- 数据对齐和预处理
- H5文件读写支持

### 评估与可视化
- `Evaluation`: 计算ATE/APE/RPE等评估指标
- `CDF`: 绘制累积分布函数图
- `Uncertainty`: 不确定性可视化
- `Poses`: 轨迹可视化

## 依赖包

- `numpy>=2.3.5` - 数值计算
- `pandas>=2.3.3` - 数据处理
- `torch>=2.9.1` - 深度学习框架
- `scipy>=1.16.3` - 科学计算
- `rerun-sdk>=0.27.2` - 数据可视化
- `h5py>=3.15.1` - H5文件处理
- `matplotlib>=3.10.8` - 图表绘制
- `opencv-python>=4.11.0.86` - 图像处理
- `onnx>=1.20.0` - ONNX格式支持
- `onnxruntime>=1.23.2` - ONNX运行时
- `sophuspy>=1.2.0` - SO(3)和李群李代数运算
- `fvcore>=0.1.5.post20221221` - 基础工具库

## 使用示例

### 1. 单模型验证

```python
from base.model import ModelLoader, DataRunner, InertialNetworkData
from base.datatype import UnitData

# 加载模型
loader = ModelLoader("/path/to/models")
model = loader.get_by_name("model.pt")

# 加载数据
data = UnitData("/path/to/unit")

# 运行预测
runner = DataRunner(data, InertialNetworkData.set_step(20))
runner.predict(model)
```

### 2. 数据集批量验证

```python
from base.datatype import DeviceDataset
from base.model import ModelLoader, DataRunner, InertialNetworkData

# 加载数据集
dataset = DeviceDataset("/path/to/dataset")

# 加载模型
loader = ModelLoader("/path/to/models")
models = loader.get_by_names(["model1.pt", "model2.pt"])

# 对每个数据单元进行验证
for data in dataset:
    runner = DataRunner(data, InertialNetworkData.set_step(10))
    runner.predict_batch(models)
```

### 3. 模型可视化

```python
from base.draw.CDF import plot_one_cdf
from base.evaluate import Evaluation

# 绘制CDF图
eval_result = Evaluation(pred_pose, gt_pose)
plot_one_cdf(eval_result.errors, save_path="results/cdf.png")
```

### 4. H5数据处理

```python
from GenerateFromH5 import H5Loader

# 加载H5文件
loader = H5Loader("/path/to/data.h5")

# 解析数据单元
imu_data, gt_data = loader._parse_unit(unit_group, remove_bias=True)
```

## 开发指南

详细的开发指南请参考 [使用指南](docs/USAGE.md)。

### 添加新模型

1. 将模型文件(.pt格式)放入模型目录
2. 在命令行中使用`-m`参数指定模型名称

### 扩展数据处理

1. 在`base/datatype.py`中定义新的数据类型
2. 在`base/interpolate.py`中添加相应的插值方法
3. 更新`base/model.py`中的处理逻辑

### 添加可视化功能

1. 在`base/draw/`目录下创建新的可视化模块
2. 使用`base.rerun_ext`进行数据可视化
3. 参考现有的`CDF.py`、`Uncertainty.py`等模块

### 模型部署

1. 使用`TorchScript.py`将模型转换为TorchScript格式
2. 使用`TorchScript2Android.py`将模型部署到Android平台
3. 参考文档了解详细的转换流程

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 联系方式

如有问题或建议，请通过GitHub Issues联系。
