import torch
from torch import nn
from unet.unet_model import UNet
class CNNModel(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        # self.model=nn.Sequential(
        #     nn.Conv2d(in_channels,out_channels,3,1,1),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channels,out_channels,3,1,1),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channels,out_channels,3,1,1),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channels,out_channels,3,1,1)
        # )
        self.model = UNet(in_channels)
    
    def forward(self,x):
        return self.model(x)
