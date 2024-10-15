import torch
from torch import nn
from model.ECAFormer import ECAFormer
from torch.nn import functional as F
from GRL import GradientReversalLayer
class ModelAPI(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.Dis_modality=nn.Sequential(
            GradientReversalLayer(),
            nn.Conv2d(in_channels,in_channels,3,2,1),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels,3,2,1),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels,3,2,1),
            nn.AdaptiveAvgPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(in_channels*4,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        self.model = ECAFormer(stage=1,n_feat=128, num_blocks=[1, 2, 2],in_channels=in_channels,out_channels=out_channels)
    
    def forward(self,x):
        b,c,h,w = x.shape
        x = F.interpolate(x,(128,128))
        x,instance,modality = self.model(x)
        return F.interpolate(x,(h,w)), instance, modality
    
    def getRepresentation(self,x):
        b,c,h,w = x.shape
        x = F.interpolate(x,(128,128))
        instance,modality = self.model.getRepresentation(x)
        return instance, modality

    def DisModality(self,x):
        return self.Dis_modality(x)
