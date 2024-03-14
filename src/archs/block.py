import torch 
import torch.nn as nn

class Conv3D(nn.Module):
    def __init__(self,
                in_channels ,
                out_channels,
                kernel_size = 3,
                stride = 1,
                padding = "same",activation= "relu"):
        super(Conv3D,self).__init__()
        self.conv = nn.Conv3d(
            in_channels = in_channels,
            stride = stride,
            out_channels = out_channels,
            kernel_size = kernel_size,
            padding  = padding,
            bias = False
        )
        self.activation = activation
        self.batch_norm = nn.BatchNorm3d(out_channels)
    def forward(self,data):
        if self.activation == "relu":
            return nn.functional.relu(self.batch_norm(self.conv(data)))
        elif self.activation is None:
            return self.batch_norm(self.conv(data))
        raise Exception 
        
class ResBlock3D(nn.Module):
        def __init__(self,
                    in_channels,
                    out_channels,
                    kernel_size = 3,
                    ):
            super(ResBlock3D,self).__init__()
            self.conv1 = Conv3D(in_channels,out_channels,kernel_size,stride = 1 ,padding  = "same",activation = "relu")
            self.conv2 = Conv3D(out_channels,out_channels,kernel_size,stride = 1 ,padding  = "same",activation = None)
            self.residual_brach  = nn.Sequential()
            if in_channels != out_channels:
                self.residual_brach = Conv3D(in_channels,out_channels,kernel_size,activation=None)
        def forward(self,data):
            out = self.conv1(data)
            out = self.conv2(out) 
            return nn.functional.relu(out)
            
if __name__ == "__main__":
    # input_tensor = torch.randn(2,20,10,56,56)
    # block = Conv3D(20,10,3,1)
    input_tensor = torch.randn(2,20,10,56,56)
    block = ResBlock3D(20,10,3)
    block(input_tensor)
    import pdb;pdb.set_trace()

    