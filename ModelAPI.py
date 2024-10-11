import torch
from torch import nn
from model.ECAFormer import ECAFormer
from torch.nn import functional as F

class ModelAPI(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.Embed_instance=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,2,1),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels,3,2,1),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels,3,2,1),
            nn.AdaptiveAvgPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(in_channels*4,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )
        self.Embed_modality=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,2,1),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels,3,2,1),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels,3,2,1),
            nn.AdaptiveAvgPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(in_channels*4,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )
        self.model = ECAFormer(stage=1,n_feat=128, num_blocks=[1, 2, 2],in_channels=in_channels,out_channels=out_channels)
    
    def forward(self,x):
        b,c,h,w = x.shape
        x = F.interpolate(x,(128,128))
        x,instance,modality = self.model(x)
        return F.interpolate(x,(h,w)), F.interpolate(instance,(h,w)), F.interpolate(modality,(h,w))
    
    def getRepresentation(self,x):
        b,c,h,w = x.shape
        x = F.interpolate(x,(128,128))
        instance,modality = self.model.getRepresentation(x)
        return instance, modality

    def getEmbedding(self,x,type):
        if type=='modality':
            return self.Embed_modality(x)
        elif type=='instance':
            return self.Embed_instance(x)
        else:
            raise NotImplementedError
        
    def getEmbeddings(self,instance,modality):
        return self.getEmbedding(instance,'instance'),self.getEmbedding(modality,'modality')