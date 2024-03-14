import torch.nn as nn
from src.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy,self).__init__()
        self.main = nn.CrossEntropyLoss()
    def forward(self,y_predict,y):
        #use soft max before 
        return self.main(y_predict,y)