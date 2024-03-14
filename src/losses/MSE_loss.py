from src.utils.registry import LOSS_REGISTRY
import torch.nn as nn
import torch


@LOSS_REGISTRY.register()
class MSE(nn.Module):
    def __init__(self):
        super(MSE,self).__init__()
        self.main = nn.MSELoss()
    def forward(self,y_predict,y):
        return self.main(y_predict,y)
