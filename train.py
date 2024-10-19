import torch
from config import Config
import matplotlib.pyplot as plt
from torch import nn
import torch.utils
from tqdm import tqdm
import numpy as np
from ModelAPI import ModelAPI
import time
from dataset.T1TauDataset import T1TauDataset
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure
from loss.NTXentLoss import NTXentLossBatch
from loss.MaxCosSimLoss import MaximizeCosineDistanceLoss
from dataset.BrainPostProcess import BrainPostProcess
import torch.nn.functional as F
from CR import ContrastLoss
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ssim_metric = StructuralSimilarityIndexMeasure()
ssim_metric.to(device)
infoNceLoss = NTXentLossBatch()
MaxCosSimLoss = MaximizeCosineDistanceLoss()
brainPostProcess = BrainPostProcess()
brainPostProcess.to(device)
CR_loss = ContrastLoss(ablation=True)
CR_loss.to(device)


def pred_one_epoch(epoch,model:nn.Module,dataloader:torch.utils.data.DataLoader,optimizer:torch.optim,criterion:torch.nn,train=True,use_consistency=False):
    losses = []
    dataloader = dataloader
    ssims = []
    for iter_num, (data,labels) in enumerate(dataloader):
        logs = {}
        start_time = time.time()
        data,labels = data.to(device),labels.to(device)
        pred,instance_x,modality_x = model(data)
        loss = criterion(pred,labels)
        logs.update({'l1_loss':loss.item()})
        cr_loss = CR_loss.get_loss(pred,labels,data)
        logs.update({'CR_loss':cr_loss.item()})
        loss+=cr_loss
        if use_consistency:
            instance_y,modality_y = model.getRepresentation(labels)
            instance_yhat,modality_yhat = model.getRepresentation(pred)
            x_modality_label = model.DisModality(modality_x,with_grl=False)
            y_modality_label = model.DisModality(modality_y,with_grl=False)
            yhat_modality_label = model.DisModality(modality_yhat,with_grl=True)
            x_mod_dis_loss = F.binary_cross_entropy_with_logits(x_modality_label,torch.zeros_like(x_modality_label))
            y_mod_dis_loss = F.binary_cross_entropy_with_logits(y_modality_label,torch.ones_like(y_modality_label))
            yhat_mod_dis_loss = F.binary_cross_entropy_with_logits(yhat_modality_label,torch.zeros_like(yhat_modality_label))
            logs.update({'dis_x':x_mod_dis_loss.item()})
            logs.update({'dis_y':y_mod_dis_loss.item()})
            logs.update({'dis_yhat':yhat_mod_dis_loss.item()})
            loss+=x_mod_dis_loss*0.005
            loss+=y_mod_dis_loss*0.005
            loss+=yhat_mod_dis_loss*0.001
            
            ins_con_loss = torch.tensor(0.,device=device)
            ins_con_loss += infoNceLoss(instance_x.flatten(1),instance_y.flatten(1))
            ins_con_loss += infoNceLoss(instance_y.flatten(1),instance_yhat.flatten(1))
            ins_con_loss += infoNceLoss(instance_x.flatten(1),instance_yhat.flatten(1))
            ins_con_loss/=3.
            logs.update({'ins_con_loss':ins_con_loss.item()})
            loss+=ins_con_loss*0.01

        logs.update({'cost_time':time.time()-start_time})
        optimizer.zero_grad()
        loss.backward()
        if train:
            optimizer.step()
        losses.append(loss.item())
        prefix = 'Train' if train else 'Test'
        pred= brainPostProcess(pred)
        for batch_id in range(pred.shape[0]):
            ssims.append(ssim_metric(pred[batch_id].unsqueeze(0), labels[batch_id].unsqueeze(0)).item())
        if (iter_num+1)%20==0:
            print(f"{prefix}:: Epoch: {epoch} | iter{iter_num}/{len(dataloader)} | Loss: {np.mean(losses)} | SSIM: {np.mean(ssims)} \n Logs : {logs}\n")
    return np.mean(ssims)
    
if __name__ == '__main__':
    opt =Config()
    model = ModelAPI(91,91)
    #model.load_state_dict(torch.load('/home/cavin/workspace/PetTauCVPR/weights/eca_ssim0.8158.pth'))
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
    criterion = nn.L1Loss()
    trainDataset = opt.dataset(train=True)
    trainDataLoader = DataLoader(trainDataset,batch_size=2,shuffle=True)
    testDataset = opt.dataset(train=False)
    testDataLoader = DataLoader(testDataset,batch_size=1)
    for i in range(5000):
        model.train()
        pred_one_epoch(i,model,trainDataLoader,optimizer,criterion,use_consistency=True)
        if (i+1)%20==0:
            model.eval()
            ssim = pred_one_epoch(i,model,testDataLoader,optimizer,criterion,train=False,use_consistency=False)
            print(ssim)
            torch.save(model.state_dict(),f'./weights/eca_ssim{ssim:.4f}.pth')
