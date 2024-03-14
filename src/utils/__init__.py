from src.utils.logger import logger
from src.utils.registry import *
import yaml

def read_yaml(yaml_path):
    with open(yaml_path,"r") as f:
        config = yaml.safe_load(f)
    return config