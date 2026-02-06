"""
使用pickle实现对象的保存和读取
"""

import pickle
from pathlib import Path
from typing import Any


class Obj:
    @staticmethod
    def save(obj: Any, path: Path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)
            print(f"Saved object to {path}")

    @staticmethod
    def load(path: Path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
            print(f"Loaded object from {path}")
            return obj

    @staticmethod
    def exist(path: Path):
        return path.exists()


class ObjSaver:
    @property
    def _obj_name(self):
        return f"{self.__class__.__name__}.pkl"

    def save(self, path):
        Obj.save(self, path)

    @staticmethod
    def load(path: Path):
        return Obj.load(path)
