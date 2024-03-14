
import os
import os.path as osp
import importlib
from src.utils import *

folder_path = osp.dirname(osp.abspath(__file__))
file_paths = [file_path.split(".")[0] for file_path in os.listdir(folder_path) if file_path.endswith("_model.py")]

_module = [importlib.import_module(f"src.models.{file_path}") for file_path in file_paths]


def get_model(opt_model:dict):
    name = opt_model['model']['type']
    model = MODEL_REGISTRY.get(name)(opt_model)
    return model 