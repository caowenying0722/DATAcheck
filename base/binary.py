import importlib
import importlib.util
import sys
from pathlib import Path

from base.datatype import UnitData


class SensorFusion:
    TARGET = "SensorFusionAndroid"

    def __init__(self, project_path: str | Path):
        self.project_path = Path(project_path)
        self.configs_path = self.project_path / "configs"
        self.build_path = self.project_path / "build"

        self.run_py = self.project_path / "Run.py"
        spec = importlib.util.spec_from_file_location("Run", self.run_py.as_posix())
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot find module at {self.run_py.as_posix()}")

        # 加载py文件为模块
        module = importlib.util.module_from_spec(spec)
        sys.modules["Run"] = module
        spec.loader.exec_module(module)
        self.runner = module.main

    def run(
        self,
        unit: Path,
        *,
        model: Path | None = None,
        config_name: str,
        use_build: bool = False,
    ):
        config_path = self.configs_path / f"{config_name}.yaml"
        return self.runner(
            self.TARGET,
            unit,
            config_path,
            model,
            build_dir=self.build_path.as_posix(),
            use_build=use_build,
        )

    def unit_run(
        self,
        ud: UnitData,
        model: Path | None = None,
        config_name: str | None = None,
        use_build: bool = False,
    ):
        config_name = ud.device_name if config_name is None else config_name
        return self.run(
            ud.base_dir, model=model, config_name=config_name, use_build=use_build
        )
