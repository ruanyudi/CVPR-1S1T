import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss

    def get_loss(self,a,p,n):
        b,c,h,w = n.shape
        loss = []
        for batch_idx in range(b):
            a_s = []
            p_s = []
            n_s = []
            for i in range(c):
                a_single=torch.cat([a[batch_idx,i,:,:].unsqueeze(0) for i in range(3)],0).unsqueeze(0)
                p_single=torch.cat([p[batch_idx,i,:,:].unsqueeze(0) for i in range(3)],0).unsqueeze(0)
                n_single=torch.cat([n[batch_idx,i,:,:].unsqueeze(0) for i in range(3)],0).unsqueeze(0)
                a_s.append(a_single)
                p_s.append(p_single)
                n_s.append(n_single)
            a_s=torch.cat(a_s,0)
            p_s=torch.cat(p_s,0)
            n_s=torch.cat(n_s,0)
            loss.append(self.forward(a_s,p_s,n_s))
        return np.mean(loss)

if __name__ == '__main__':
    dummy_a = torch.randn((4,91,91,109))
    dummy_p = torch.randn((4, 91, 91, 109))
    dummy_n = torch.randn((4, 91, 91, 109))
    loss = ContrastLoss(ablation=True).get_loss(dummy_a,dummy_p,dummy_n)
    print(loss)
    loss = ContrastLoss(ablation=False).get_loss(dummy_a, dummy_p, dummy_n)
    print(loss)
    # print(ContrastLoss()(dummy_a,dummy_p,dummy_n))