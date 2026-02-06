import time
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

import base.rerun_ext as rre

from . import device
from .datatype import ExtUnitData, ImuData, Pose, PosesData, UnitData

NetworkOutput: TypeAlias = tuple[NDArray, NDArray]


class InertialNetwork:
    model: torch.nn.Module

    def __init__(
        self,
        model_path: Path | str,
        input_shape: tuple[int, ...] | None = None,
    ):
        self.model_path = Path(model_path)
        self.name = self.model_path.name.split(".")[0]
        self.device = device.DefaultDevice
        self.device_name = str(self.device)
        self.model = torch.jit.load(self.model_path, map_location=self.device)

        if input_shape is not None:
            self.input_shape = input_shape
        else:
            self.input_shape = (1, 6, 200)
        self.model.eval()
        print(f"Model: {self.model_path.name} load success.")

        self.warmup()
        print("Warmup complete.")

    def warmup(self):
        block = torch.randn(self.input_shape, dtype=torch.float32, device=self.device)
        self.model(block)

    def predict(self, block: NDArray) -> NetworkOutput:
        if self.input_shape:
            assert block.shape == self.input_shape, (
                f"Input shape mismatch: {block.shape} != {self.input_shape}"
            )
        inputs = torch.as_tensor(block, dtype=torch.float32, device=self.device)
        output = self.model(inputs)
        meas: NDArray = output[0].cpu().detach().numpy().flatten()
        meas_cov: NDArray = output[1].cpu().detach().numpy().flatten()
        return meas, meas_cov

    def predict_with_time(self, block: NDArray) -> tuple[NetworkOutput, float]:
        start_time = time.perf_counter()
        if self.input_shape:
            assert block.shape == self.input_shape, (
                f"Input shape mismatch: {block.shape} != {self.input_shape}"
            )
        inputs = torch.as_tensor(block, dtype=torch.float32, device=self.device)
        output = self.model(inputs)
        meas: NDArray = output[0].cpu().detach().numpy().flatten()
        meas_cov: NDArray = output[1].cpu().detach().numpy().flatten()
        end_time = time.perf_counter()
        return (meas, meas_cov), end_time - start_time

    def reset(self):
        self.model.eval()


class MemoryInertialNetwork(InertialNetwork):
    Memory = tuple[torch.Tensor, torch.Tensor]
    hn: torch.Tensor = torch.zeros((1, 1, 256))
    cn: torch.Tensor = torch.zeros((1, 1, 256))

    def __init__(
        self,
        model_path: Path | str,
        input_shape: tuple[int, ...] | None = None,
        polling_rate=10,
    ):
        super().__init__(model_path, input_shape)
        self.polling_rate = polling_rate
        self.idx = 0
        self.memories = []

    def get_memory(self) -> Memory:
        if len(self.memories) <= 10:
            return (self.hn, self.cn)
        else:
            return self.memories[self.idx - self.polling_rate]

    def set_memory(self, memory: Memory):
        self.memories.append(memory)
        self.idx += 1

    def predict(self, block: NDArray) -> NetworkOutput:
        if self.input_shape:
            assert block.shape == self.input_shape, (
                f"Input shape mismatch: {block.shape} != {self.input_shape}"
            )
        inputs = torch.as_tensor(block, dtype=torch.float32, device=self.device)
        memory = self.get_memory()
        output = self.model(inputs, *memory)

        meas: NDArray = output[0].cpu().detach().numpy().flatten()
        meas_cov: NDArray = output[1].cpu().detach().numpy().flatten()

        if len(output) > 2:
            hn = output[2]
            cn = output[3]
            self.set_memory((hn, cn))

        return meas, meas_cov

    def reset(self):
        self.model.eval()
        self.idx = 0
        self.memories = []


class ModelLoader:
    _suffix = ".pt"
    models: list[Path] = []

    def __init__(self, base_dir: Path | str):
        path = Path(base_dir)
        self.models = [
            it
            for it in path.iterdir()
            if it.name.endswith(self._suffix) and it.is_file()
        ]

    def __iter__(self):
        for model_path in self.models:
            yield InertialNetwork(model_path)

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return [InertialNetwork(model_path) for model_path in self.models[index]]
        return InertialNetwork(self.models[index])

    def __len__(self):
        return self.models.__len__()

    def search_by_name(self, name: str):
        for model_path in self.models:
            model_name: str = model_path.name
            if name in model_name:
                return model_path
        raise ValueError(f"Model {name} not found.")

    def get_by_names(self, names: list[str]):
        return [self.get_by_name(name) for name in names]

    def get_by_name(self, name: str):
        model_path = self.search_by_name(name)
        if "lstm" in model_path.name:
            return MemoryInertialNetwork(model_path)
        return InertialNetwork(model_path)

    def get_networks(self):
        return [InertialNetwork(model_path) for model_path in self.models]


class NetworkResult:
    step: int
    rate: int
    t_start_us: int
    # 测量结果
    meas_list: list[NDArray]
    meas_cov_list: list[NDArray]

    gt_list: list[NDArray]  # 20Hz速度列表 rate / step
    err_list: list  # 误差
    eval_t_list: list[float]  # 模型推理耗时

    pose: Pose
    pose_list: list[Pose]  # 位置结果

    def __init__(
        self,
        tag: str = "",
        init_positon: NDArray = np.zeros(3),
        *,
        step: int = 10,
        rate: int = 200,
        using_rerun: bool = True,
        network_device_name: str = "",
    ):
        self.tag = tag
        self.step = step
        self.rate = rate
        self.using_rerun = using_rerun
        self.interval_us = int(1e6 * step / rate)

        self.gt_list = []
        self.err_list = []
        self.eval_t_list = []
        self.network_device_name = network_device_name

        self.meas_list = []
        self.meas_cov_list = []
        self.positon = init_positon.copy()
        self.pose_list = []
        self.path = []

        if self.using_rerun:
            rre.log_coordinate(
                f"/world/{self.tag}",
                length=1,
                labels=[f"Network{self.tag}"],
                show_labels=False,
            )

    def __len__(self):
        return len(self.meas_list)

    def __getitem__(self, index: int):
        return self.meas_list[index], self.meas_cov_list[index]

    @property
    def poses(self):
        return PosesData.from_list(self.pose_list)

    def add(
        self,
        output: NetworkOutput,
        ref_pose: Pose = Pose.identity(),
        ref_disp: NDArray | None = None,
        eval_t: float = 0.0,
    ):
        self.meas_list.append(output[0])
        self.meas_cov_list.append(output[1])
        self.eval_t_list.append(eval_t)

        self.positon += output[0] * self.interval_us / 1e6
        pose = ref_pose.copy()
        pose.p = self.positon.copy()

        self.pose_list.append(pose)
        self.path.append(pose.p)

        if self.using_rerun and ref_pose.t_us > 0:
            rre.log_network_pose(ref_pose.t_us, pose, self.path, tag=self.tag)

        # 统计误差
        if ref_disp is not None:
            self.gt_list.append(ref_disp)
            err = np.linalg.norm(output[0] - ref_disp)
            self.err_list.append(err)

        return ref_pose


class InertialNetworkData:
    """
    惯性网络数据处理类

    该类用于处理IMU数据，为神经网络预测提供数据块。主要功能包括：
    - 数据预处理（移除重力、时间范围裁剪）
    - 生成固定大小的数据块用于网络输入
    - 支持数据旋转以适应不同坐标系
    - 提供批量预测功能

    Attributes:
        step (int): 数据块之间的步长，默认为80
        rate (int): IMU数据采样率，默认为200Hz
        rm_g (bool): 是否移除重力加速度，默认为False
        _idx_range (slice): 时间范围索引切片
        imu_block (NDArray[np.float32]): 处理后的IMU数据块，包含陀螺仪和加速度计数据

    Note:
        输入的IMU数据应处于全局坐标系（global frame）
    """

    step = 80
    rate = 200
    rm_g: bool = False
    _idx_range: slice = slice(None, None, None)
    imu_block: NDArray[np.float32]

    using_rerun: bool = True
    rotate_heading: bool = False

    @classmethod
    def set_step(cls, step: int):
        cls.step = step
        return cls

    @classmethod
    def set_rate(cls, rate: int):
        cls.rate = rate
        return cls

    @classmethod
    def no_rerun(cls):
        cls.using_rerun = False
        return cls

    @classmethod
    def remove_gravity(cls):
        cls.rm_g = True
        return cls

    @classmethod
    def set_time_range(cls, time_range: tuple[float, float]):
        ts, te = time_range
        start_idx = int(ts * cls.rate) if ts is not None else None
        end_idx = int(te * cls.rate) if te is not None else None

        cls._idx_range = slice(start_idx, end_idx)
        return cls

    @classmethod
    def heading(cls, rotate_heading: bool = True):
        cls.rotate_heading = rotate_heading
        return cls

    def __init__(
        self,
        world_imu_data: ImuData,
        *,
        remove_gravity: bool | None = None,
    ) -> None:
        assert world_imu_data.frame == "global", "Imu not in global frame."
        if remove_gravity is not None:
            self.rm_g = remove_gravity
        if self.rm_g:
            world_imu_data.acce -= np.array([0, 0, 9.81])

        self.world_imu_data = world_imu_data[self._idx_range]
        self.imu_block = np.hstack([self.world_imu_data.gyro, self.world_imu_data.acce])

        self.shape = (1, 6, self.rate)

    def get_block(self):
        self.bc = 0
        while self.bc + self.rate < len(self.imu_block):
            block = self.imu_block[self.bc : self.bc + self.rate]
            yield block.T.reshape(self.shape)
            self.bc += self.step

    def get_block_idx(self):
        self.bc = 0
        while self.bc + self.rate < len(self.imu_block):
            block = self.imu_block[self.bc : self.bc + self.rate]  # 200 x 6
            yield (self.bc, block.T.reshape(self.shape))
            self.bc += self.step

    def rotate_block(self, block: NDArray, rot: Rotation):
        """
        block : [1, 6, 200], gyro(3), acce(3)
        """
        gyro_block = block[0, 0:3, :].T
        acce_block = block[0, 3:6, :].T
        gyro_block = rot.apply(gyro_block)
        acce_block = rot.apply(acce_block)
        return np.hstack([gyro_block, acce_block]).T.reshape(self.shape)

    def predict_using_rot(
        self, network: InertialNetwork, ref_poses: PosesData, degrees: list = [0]
    ):
        results = [
            NetworkResult(
                f"{network.name}_{deg}",
                ref_poses.get_pose(0).p,
                step=self.step,
                rate=self.rate,
            )
            for deg in degrees
        ]

        for idx, block in self.get_block_idx():
            ref_pose = ref_poses.get_pose(idx)
            for i, deg in enumerate(degrees):
                # 旋转任意角度
                yaw_rot = Rotation.from_rotvec([0, 0, deg], degrees=True)
                block = self.rotate_block(block, yaw_rot)
                # 获取网络结果， 统计耗时
                (meas, meas_cov), eval_t = network.predict_with_time(block)

                # 反向旋转
                meas = yaw_rot.inv().apply(meas)
                meas_cov = yaw_rot.inv().apply(meas_cov)

                _pose = results[i].add((meas, meas_cov), ref_pose, eval_t=eval_t)
                print(f"{network.name}_{deg}-{idx}: {_pose.p}")
        return results

    def predict_usings(
        self,
        networks: list[InertialNetwork],
        ref_poses: PosesData,
        test_scale: bool = False,
        test_heading: bool = False,
    ):
        """
        批量预测是否
        """
        results = [
            NetworkResult(
                model.name,
                ref_poses.get_pose(0).p,
                step=self.step,
                rate=self.rate,
                using_rerun=self.using_rerun,
                network_device_name=model.device_name,
            )
            for model in networks
        ]

        for idx, block in self.get_block_idx():
            s_pose = ref_poses.get_pose(idx)
            e_pose = ref_poses.get_pose(idx + self.rate)
            disp = e_pose.p - s_pose.p

            # 对齐行走方向
            if self.rotate_heading:
                yaw_rot = s_pose.get_yaw_pose().rot
                block = self.rotate_block(block, yaw_rot)
            else:
                yaw_rot = Rotation.identity()

            for i, net in enumerate(networks):
                (meas, meas_cov), eval_t = net.predict_with_time(block)

                # 对齐行走方向
                if self.rotate_heading:
                    meas = yaw_rot.inv().apply(meas)
                    meas_cov = yaw_rot.inv().apply(meas_cov)

                # 测试 输出长度的准确性
                if test_scale:
                    meas = np.linalg.norm(meas) * disp / np.linalg.norm(disp)
                # 测试 输出角度的准确性
                if test_heading:
                    meas = np.linalg.norm(disp) * meas / np.linalg.norm(meas)

                # fix_rot = Rotation.from_rotvec([0, 0, 8.11], degrees=True)
                # meas = fix_rot.apply(meas)

                # 获取结果
                _pose = results[i].add((meas, meas_cov), s_pose, disp, eval_t=eval_t)

        return results


class DataRunner:
    # 保持尺度不变，真值方向
    test_scale: bool = False
    # 保持方向不变，真值尺度
    test_heading: bool = False

    def __init__(
        self,
        ud: UnitData,
        Data: type[InertialNetworkData] = InertialNetworkData,
        *,
        time_range: tuple[float | None, float | None] = (None, None),
        has_init_rerun: bool = False,
        using_gt: bool = True,
    ):
        self.ud = ud
        assert len(self.ud.gt_data) == len(self.ud.imu_data), (
            f"GT and IMU data length mismatch: {len(self.ud.gt_data)} != {len(self.ud.imu_data)}"
        )
        self.gt_data = ud.gt_data.get_time_range(time_range)
        self.imu_data = ud.imu_data.get_time_range(time_range)

        # 变换到global
        world_imu_gt = self.imu_data.transform(self.gt_data.rots if using_gt else None)
        self.in_data = Data(world_imu_gt)

        # 获取 gt_data 的起始位置
        assert len(self.gt_data) > 0, f"{self.gt_data}"
        self.init_pose = Pose.from_transform(self.gt_data.ps[0])

        if not has_init_rerun and self.in_data.using_rerun:
            # rre.rerun_init(ud.name, imu_view_tags=["GT", "Raw"])
            rre.RerunView().add_imu_view(
                tags=["GT", "Raw"],
            ).add_cdf_view().send(ud.name)
            rre.send_pose_data(self.gt_data, "GT", color=[192, 72, 72])
            rre.send_imu_data(self.imu_data, tag="Raw")
            rre.send_imu_data(world_imu_gt, tag="GT")

            # 如果是ext
            if self.ud.using_ext:
                assert isinstance(self.ud, ExtUnitData)

                fusion_data = self.ud.fusion_data.get_time_range(time_range)
                fusion_data.ps = fusion_data.ps - fusion_data.ps[0] + self.gt_data.ps[0]
                rre.send_pose_data(fusion_data, "Fusion")

    def predict_rot(self, network: InertialNetwork, degrees: list = [0]):
        return self.in_data.predict_using_rot(network, self.gt_data, degrees)

    def predict_batch(self, networks: list[InertialNetwork]):
        results = self.in_data.predict_usings(
            networks,
            self.gt_data,
            self.test_scale,
            self.test_heading,
        )
        return results
