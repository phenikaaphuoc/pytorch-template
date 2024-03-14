
import os
import os.path as osp
import importlib
from src.utils import *

folder_path = osp.dirname(osp.abspath(__file__))
file_paths = [file_path.split(".")[0] for file_path in os.listdir(folder_path) if file_path.endswith("_dataset.py")]
_module = [importlib.import_module(f"src.data.{file_path}") for file_path in file_paths]


def get_dataset(opt:dict):
    name = opt['type']
    opt.pop('type',None)
    data = DATASET_REGISTRY.get(name)(**opt)
    return data