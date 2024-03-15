from src.utils import *
from src.utils.visualize import Visualizor
from src.models import get_model 
import argparse
import torch
from src.utils.data import get_dataloader_helper
import numpy as np
from src.utils.validation import Validation

def main(args):
    
    config = read_yaml(args.config_file)
    model = get_model(config)
    train_data_loader_list,val_data_loader_list = get_dataloader_helper(config["data"])
    count = 0
    validataion  = Validation(config)
    logger.info("Training start")
    for train_dataloader in train_data_loader_list:
        if count > config['train_parameter']['total_iter']:
            logger.info("Trainning finish")
            break
        for i , (input,target) in enumerate(train_dataloader):
            count+=1
            model.feedData((input,target))
            model.optimize()
            if count > config['train_parameter']['total_iter']:
                logger.info("Trainning finish")
                break
            if count % config["train_parameter"]["frequent"] == 0:
                logger.info(f"Loss in iter {count} : {model.get_current_loss()}")
            if config["val_parameter"]["frequent"]:
                validataion.validate(model,val_data_loader_list,count)
                          


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument('--config_file', type=str, default=r"src/configs/hpc_train_config.yaml", help='path to config file')
    args = parser.parse_args()
    main(args)
