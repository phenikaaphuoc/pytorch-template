
import os
import os.path as osp
import importlib
from src.utils import *

folder_path = osp.dirname(osp.abspath(__file__))
file_paths = [file_path.split(".")[0] for file_path in os.listdir(folder_path) if file_path.endswith("_arch.py")]
_module = [importlib.import_module(f"src.archs.{file_path}") for file_path in file_paths]


def get_arch(opt_arch:dict):
    name = opt_arch['type']
    arch = ARCH_REGISTRY.get(name)(opt_arch)
    return arch