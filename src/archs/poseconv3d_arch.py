from src.utils import *
import torch.nn as nn
from src.archs.block import *

@ARCH_REGISTRY.register()
class poseconv3d(nn.Module):
    def __init__(self,config):
        super(poseconv3d,self).__init__()
        
        self.main_net = nn.ModuleList()
        in_channels = 3
        for out_channels in config['in_channels'][:-1]:
            layer = [Conv3D(in_channels, out_channels, 3)]if in_channels == 3 else [ResBlock3D(in_channels, out_channels), nn.MaxPool3d(2, 2)]
            self.main_net.extend(layer)
            in_channels = out_channels
        out_channels = config['in_channels'][-1]
        self.main_net.extend(
            [
                ResBlock3D(in_channels, out_channels),
                nn.AdaptiveAvgPool3d(1)
            ]
        )     
        self.main_net = nn.Sequential(*self.main_net)
        self.linear = nn.Linear(out_channels,config['num_class'])
        
    def forward(self,data):
        out = self.main_net(data).squeeze()
        return self.linear(out)
        
if __name__ == "__main__":
  
    input = torch.randn(3,3,32,56,56)
   