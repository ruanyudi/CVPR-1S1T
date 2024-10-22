import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet.unet_model import UNet
from einops import rearrange
from fvcore.nn import FlopCountAnalysis


class RepNet(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.ModalityNet = UNet(n_channels)
        self.InstanceNet = UNet(n_channels)

    def forward(self, img):
        Modality = self.ModalityNet(img)
        Instance = self.InstanceNet(img)
        return Instance, Modality

class TriUnet(nn.Module):
    def __init__(self,in_channels=91):
        self.RepNet = RepNet(n_channels=in_channels)
        self.MixNet = UNet(in_channels*2)

    def forward(self,x):
        instance,modality = self.RepNet(x)
        return self.MixNet(torch.cat((instance,modality),dim=1))

    def getRepresentation(self, x):
        return self.RepNet(x)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TriUnet(in_channels=91).to(device)
    inputs = torch.randn((1, 91, 128, 128)).to(device)
    model.getRepresentation(inputs)
    flops = FlopCountAnalysis(model, inputs)
    n_param = sum([p.nelement() for p in model.parameters()])
    print(f'GMac:{flops.total() / (1024 * 1024 * 1024)}')
    print(f'Params:{n_param}')
    # model.load_state_dict(torch.load("./LOL-v1.pth")['params'])