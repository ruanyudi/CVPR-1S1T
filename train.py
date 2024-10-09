import torch
from torch import nn
import torch.utils
from tqdm import tqdm
import numpy as np
from ModelAPI import ModelAPI
import time
from T1TauDataset import T1TauDataset
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure
from NTXentLoss import NTXentLossBatch
from MaxCosSimLoss import MaximizeCosineDistanceLoss
from BrainPostProcess import BrainPostProcess
import torch.nn.functional as F
device = 'cuda'

ssim_metric = StructuralSimilarityIndexMeasure()
ssim_metric.to(device)
infoNceLoss = NTXentLossBatch()
MaxCosSimLoss = MaximizeCosineDistanceLoss()
brainPostProcess = BrainPostProcess()
brainPostProcess.to(device)


def pred_one_epoch(epoch,model:nn.Module,dataloader:torch.utils.data.DataLoader,optimizer:torch.optim,criterion:torch.nn,train=True,use_consistency=False):
    losses = []
    dataloader = dataloader
    ssims = []
    for iter_num, (data,labels) in enumerate(dataloader):
        logs = {}
        start_time = time.time()
        data,labels = data.to(device),labels.to(device)
        pred,instance_x,modality_x = model(data)
        pred= brainPostProcess(pred)
        loss = criterion(pred,labels)
        logs.update({'mse_loss':loss.item()})
        embed = lambda x : model.getEmbeddings(*x)
        if use_consistency:
            instance_y,modality_y = model.getRepresentation(labels)
            instance_yhat,modality_yhat = model.getRepresentation(pred)
            instance_x,modality_x = embed([instance_x,modality_x])
            instance_yhat,modality_yhat=embed([instance_yhat,modality_yhat])
            instance_y,modality_y=embed([instance_y,modality_y])

            ### instance_x instance_yhat instance_y
            instance_consistency_loss=0
            instance_consistency_loss += infoNceLoss(instance_x,instance_yhat)
            instance_consistency_loss += infoNceLoss(instance_x,instance_y)
            instance_consistency_loss += infoNceLoss(instance_y,instance_yhat)
            instance_consistency_loss /=3.
            logs.update({'ins_closs':instance_consistency_loss.item()})
            loss = loss + instance_consistency_loss*0.01
            ### modality_x modality_t modality_yhat
            modality_consistency_loss=0
            modality_consistency_loss+=F.mse_loss(modality_y,modality_yhat)
            modality_consistency_loss+=MaxCosSimLoss(modality_x,modality_y)
            modality_consistency_loss+=MaxCosSimLoss(modality_yhat,modality_x)
            logs.update({'mod_closs':modality_consistency_loss.item()})
            loss = loss+modality_consistency_loss


        # logs.update({'cost_time':time.time()-start_time})
        optimizer.zero_grad()
        loss.backward()
        if train:
            optimizer.step()
        losses.append(loss.item())
        prefix = 'Train' if train else 'Test'
        ssims.append(ssim_metric(pred,labels).item())
        if (iter_num+1)%50==0:
            print(f"{prefix}:: Epoch: {epoch} | iter{iter_num}/{len(dataloader)} | Loss: {np.mean(losses)} | SSIM: {np.mean(ssims)} \n Logs : {logs}\n")
    return np.mean(ssims)
    
if __name__ == '__main__':
    model = ModelAPI(91,91)
    model.load_state_dict(torch.load('/home/cavin/workspace/PetTauCVPR/weights/eca_ssim0.8980.pth'))
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
    criterion = nn.L1Loss()
    trainDataset = T1TauDataset(train=True)
    trainDataLoader = DataLoader(trainDataset,batch_size=4,shuffle=True)
    testDataset = T1TauDataset(train=False)
    testDataLoader = DataLoader(testDataset,batch_size=4)
    for i in range(5000):
        model.train()
        pred_one_epoch(i,model,trainDataLoader,optimizer,criterion,use_consistency=False)
        if (i+1)%20==0:
            model.eval()
            ssim = pred_one_epoch(i,model,testDataLoader,optimizer,criterion,train=False,use_consistency=False)
            print(ssim)
            torch.save(model.state_dict(),f'./weights/eca_ssim{ssim:.4f}.pth')
