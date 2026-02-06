from itertools import accumulate

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from numpy.typing import NDArray

from .datatype import ImuData, Pose, PosesData


class RerunView:
    def __init__(self):
        self.views = []

    def add_imu_view(
        self,
        visible: bool = False,
        tags: list = [],
        range: tuple[float, float] = (-15.0, 15.0),
    ):
        XYZ_AXIS_NAMES = ["x", "y", "z"]
        XYZ_AXIS_COLORS = [[(231, 76, 60), (39, 174, 96), (52, 120, 219)]]

        time_range_start = rrb.TimeRangeBoundary.cursor_relative(seconds=range[0])
        time_range_end = rrb.TimeRangeBoundary.cursor_relative(seconds=range[1])
        gyro_views = [
            rrb.TimeSeriesView(
                origin=f"gyroscope_{tag}",
                name=f"Gyroscope_{tag}",
                overrides={
                    # type: ignore[arg-type]
                    f"/gyroscope_{tag}": rr.SeriesLines.from_fields(
                        names=XYZ_AXIS_NAMES, colors=XYZ_AXIS_COLORS
                    ),
                },
                axis_x=rrb.archetypes.TimeAxis(
                    view_range=rrb.TimeRange(start=time_range_start, end=time_range_end)
                ),
            )
            for tag in tags
        ]
        acce_views = [
            rrb.TimeSeriesView(
                origin=f"accelerometer_{tag}",
                name=f"Accelerometer_{tag}",
                overrides={
                    # type: ignore[arg-type]
                    f"/accelerometer_{tag}": rr.SeriesLines.from_fields(
                        names=XYZ_AXIS_NAMES, colors=XYZ_AXIS_COLORS
                    ),
                },
                axis_x=rrb.archetypes.TimeAxis(
                    view_range=rrb.TimeRange(start=time_range_start, end=time_range_end)
                ),
            )
            for tag in tags
        ]

        magn_views = [
            rrb.TimeSeriesView(
                origin=f"magnetometer_{tag}",
                name=f"Magnetometer_{tag}",
                overrides={
                    # type: ignore[arg-type]
                    f"/magnetometer_{tag}": rr.SeriesLines.from_fields(
                        names=XYZ_AXIS_NAMES, colors=XYZ_AXIS_COLORS
                    ),
                },
                axis_x=rrb.archetypes.TimeAxis(
                    view_range=rrb.TimeRange(start=time_range_start, end=time_range_end)
                ),
            )
            for tag in tags
        ]

        imu_view = rrb.Vertical(*gyro_views, *acce_views, *magn_views, visible=visible)

        self.views.append(imu_view)
        return self

    def add_spatial_view(
        self,
        eye_at_entity: str = "/world/Groundtruth",
    ):
        spatial_view = rrb.Spatial3DView(
            origin="/",
            name="World position",
            spatial_information=rrb.SpatialInformation(show_axes=True),
            eye_controls=rrb.EyeControls3D(
                kind="Orbital", tracking_entity=eye_at_entity
            ),
        )
        self.views.append(spatial_view)
        return self

    def add_cdf_view(
        self,
        visible: bool = True,
        tags: list = [],
        error_types: list = ["ATE", "RTE"],
        range: tuple[float, float] = (-15.0, 15.0),
    ):
        """
        添加 CDF 视图，用于显示误差的累积分布函数

        Args:
            visible: 是否可见
            tags: 数据标签列表
            error_types: 误差类型列表，如 ['ATE', 'RTE', 'APE', 'RPE']
            range: 时间轴范围（相对于光标）
        """
        # 为每个误差类型创建颜色映射
        ERROR_TYPE_COLORS = {
            "APE": [(231, 76, 60)],  # 红色
            "ATE": [(39, 174, 96)],  # 绿色
            "RPE": [(52, 120, 219)],  # 蓝色
            "RTE": [(255, 152, 0)],  # 橙色
        }

        time_range_start = rrb.TimeRangeBoundary.cursor_relative(seconds=range[0])
        time_range_end = rrb.TimeRangeBoundary.cursor_relative(seconds=range[1])

        # 为每个 tag 和 error_type 组合创建视图
        cdf_views = []
        for tag in tags:
            for error_type in error_types:
                view_name = f"CDF_{error_type}_{tag}"
                origin = f"cdf_{error_type}_{tag}"

                # 创建时间序列视图
                view = rrb.TimeSeriesView(
                    origin=origin,
                    name=view_name,
                    overrides={
                        origin: rr.SeriesLines.from_fields(
                            names=["CDF"],
                            colors=[
                                ERROR_TYPE_COLORS.get(error_type, [(128, 128, 128)])
                            ],
                        ),
                    },
                    axis_x=rrb.archetypes.TimeAxis(
                        view_range=rrb.TimeRange(
                            start=time_range_start, end=time_range_end
                        )
                    ),
                    visible=visible,
                )
                cdf_views.append(view)

        if cdf_views:
            cdf_view = rrb.Vertical(*cdf_views, visible=visible)
            self.views.append(cdf_view)

        return self

    def send(self, name: str, shares: list[float] | None = []):
        rr.init(name, spawn=True)
        blueprint = rrb.Horizontal(*self.views, column_shares=shares)
        rr.send_blueprint(blueprint)
        return self


def rerun_init(name: str):
    RerunView().add_spatial_view().send(name)


def log_coordinate(
    entity_path: str,
    view_coordinate: rr.AsComponents = rr.ViewCoordinates.RIGHT_HAND_Z_UP,
    length: float = 1.0,
    radii=None,
    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    labels=["X", "Y", "Z"],
    show_labels=None,
):
    rr.log(
        entity_path,
        view_coordinate,
        rr.Arrows3D(
            vectors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * length,
            radii=radii,
            colors=colors,
            labels=labels,
            show_labels=show_labels,
        ),
        static=True,
    )


def set_world_tf(tf: Pose):
    rr.log(
        "/world",
        rr.Transform3D(mat3x3=tf.rot.as_matrix(), translation=tf.p),
        static=True,
    )


def send_imu_data(imu_data: ImuData, tag: str = ""):
    ts_imu = rr.TimeColumn("timestamp", timestamp=imu_data.t_us * 1e-6)
    rr.send_columns(
        f"/gyroscope_{tag}",
        indexes=[ts_imu],
        columns=rr.Scalars.columns(scalars=imu_data.gyro),
    )
    rr.send_columns(
        f"/accelerometer_{tag}",
        indexes=[ts_imu],
        columns=rr.Scalars.columns(scalars=imu_data.acce),
    )
    rr.send_columns(
        f"/magnetometer_{tag}",
        indexes=[ts_imu],
        columns=rr.Scalars.columns(scalars=imu_data.magn),
    )


def send_pose_data(poses_data: PosesData, tag: str = "Pose", color=None):
    times = rr.TimeColumn("timestamp", timestamp=poses_data.t_us * 1e-6)
    ps = poses_data.ps
    qs = poses_data.rots.as_quat(canonical=True)

    log_coordinate(f"/world/{tag}", length=1, labels=[tag], show_labels=False)
    rr.send_columns(
        f"/world/{tag}",
        indexes=[times],
        columns=rr.Transform3D.columns(quaternion=qs, translation=ps),
    )

    colors = [color] if color is not None else None
    rr.log(
        f"/world/{tag}_path",
        rr.LineStrips3D(strips=[ps], labels=[tag], colors=colors, show_labels=True),
        static=True,
    )


def send_pose_data_dyn(poses_data: PosesData, tag: str = "Pose", color=None):
    times = rr.TimeColumn("timestamp", timestamp=poses_data.t_us * 1e-6)
    ps = poses_data.ps
    qs = poses_data.rots.as_quat(canonical=True)

    log_coordinate(f"/world/{tag}", length=1, labels=[tag], show_labels=False)
    rr.send_columns(
        f"/world/{tag}",
        indexes=[times],
        columns=rr.Transform3D.columns(quaternion=qs, translation=ps),
    )

    def _gen_path(ps: np.ndarray | list[np.ndarray]):
        paths = list(accumulate(ps, lambda acc, x: acc + [x], initial=[]))
        return paths[1:]

    paths = _gen_path(ps)
    tags = [tag] * len(paths) if tag is not None else None
    colors = [np.array([color])] * len(paths) if color is not None else None
    rr.send_columns(
        f"/world/{tag}_path",
        indexes=[times],
        columns=rr.LineStrips3D.columns(strips=paths, labels=tags, colors=colors),
    )


def log_network_pose(
    t_us: int, pose: Pose, path: list[NDArray] | None = None, tag: str = ""
):
    rr.set_time("timestamp", timestamp=t_us / 1e6)
    rr.log(
        f"/world/{tag}",
        rr.Transform3D(mat3x3=pose.rot.as_matrix(), translation=pose.p),
    )
    if path is not None:
        rr.log(
            f"/world/{tag}_path",
            rr.LineStrips3D(
                strips=[path],
                labels=[f"{tag}"],
                colors=[[192, 72, 192]],
                show_labels=True,
            ),
        )
