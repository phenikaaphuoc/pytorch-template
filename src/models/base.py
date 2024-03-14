import torch.nn as nn
from src.losses import get_loss
import torch
class BaseModel(nn.Module):
    def __init__(self,config):
        super(BaseModel,self).__init__()
        self.config = config
        
    def feetData(self,data):
        pass
    def setDevice(self,device:str):
        pass
    def setLoss(self):
        self.loss_fn  = get_loss(self.config)
    def setOptim(self,net:nn.Module):
        self.optim = torch.optim.Adam(net.parameters(),**(self.config["optim"]))
    def setDevice(self, device: str):
        self.device = torch.device(device)
        self.net.to(self.device)
    def optimize(self):
        pass
    