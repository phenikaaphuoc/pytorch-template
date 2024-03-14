import os
import os.path as osp
import importlib
from src.utils import *

folder_path = osp.dirname(osp.abspath(__file__))
file_paths = [file_path.split(".")[0] for file_path in os.listdir(folder_path) if file_path.endswith("_loss.py")]

_module = [importlib.import_module(f"src.losses.{file_path}") for file_path in file_paths]
def get_loss(opt:dict):
    name = opt['loss']['type']
    model = LOSS_REGISTRY.get(name)()
    return model 