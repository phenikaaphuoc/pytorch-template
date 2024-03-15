from src.utils import *
import torch.nn as nn
from src.models.base import BaseModel
from src.archs import get_arch
import torch
import numpy as np
@MODEL_REGISTRY.register()
class PoseConv3D(BaseModel):
    def __init__(self, config):
        super(PoseConv3D, self).__init__(config)
        
        self.net = get_arch(config['model']['arch'])
        logger.info(f"Load backbone type {config['model']['arch']['type']} succesfully")
        self.setDevice(config["device"])
        logger.info(f"Device : {config['device']}")
        if config.get("mode",None)== "train":
            logger.info("Mode: train")
            self.setLoss()
            self.setOptim(self.net)
            self.train_loss = []
            self.valid_loss = []
        logger.info("Load model succesfully")


    def setDevice(self, device: str):
        self.device = torch.device(device)
        self.net.to(self.device)

    def feedData(self, data: tuple):
        self.input, self.target = data
        self.input = self.input.to(self.device)
        self.target = self.target.to(self.device)

    def optimize(self):
        self.optim.zero_grad()
        #shape batch , N
        self.inference()
        loss = self.loss_fn(self.output, self.target)
        loss.backward()
        self.optim.step()
        self.train_loss.append(round(loss.detach().cpu().item(),2))
        del self.output  
        del self.input
    def get_current_loss(self):
        return self.train_loss[-1]
    def get_output(self):
        if not isinstance(self.output,np.ndarray):
          self.output = self.output.detach().cpu()
        return self.output.numpy()
    def inference(self):
        self.output = self.forward()
    def forward(self):
        return nn.functional.softmax(self.net(self.input),dim = -1)
