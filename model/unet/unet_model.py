""" Full assembly of the parts to form the complete network """
import time

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=False, scale=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64*scale))
        self.down1 = (Down(64*scale, 128*scale))
        self.down2 = (Down(128*scale, 256*scale))
        self.down3 = (Down(256*scale, 512*scale))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512*scale, 1024*scale // factor))
        self.up1 = (Up(1024*scale, 512*scale // factor, bilinear))
        self.up2 = (Up(512*scale, 256*scale // factor, bilinear))
        self.up3 = (Up(256*scale, 128*scale // factor, bilinear))
        self.up4 = (Up(128*scale, n_channels, bilinear))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

if __name__ == '__main__':
    model=UNet(n_channels=91, bilinear=True)
    dummy_input = torch.randn((1,91,91,114))
    start = time.time()
    pred = model(dummy_input)
    print(time.time() - start)
    print(pred.shape)