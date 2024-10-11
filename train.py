import torch
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
device = 'mps'
device = 'cuda' if torch.cuda.is_available() else 'mps'

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
        pred= brainPostProcess(pred)
        loss = criterion(pred,labels)
        logs.update({'mse_loss':loss.item()})
        cr_loss = CR_loss.get_loss(pred,labels,data)
        logs.update({'CR_loss':cr_loss.item()})
        loss+=cr_loss
        embed = lambda x : model.getEmbeddings(*x)
        if use_consistency:
            instance_y,modality_y = model.getRepresentation(labels)
            instance_yhat,modality_yhat = model.getRepresentation(pred)
            x_modality_label = model.DisModality(modality_x)
            y_modality_label = model.DisModality(modality_y)
            yhat_modality_label = model.DisModality(modality_yhat)


        logs.update({'cost_time':time.time()-start_time})
        optimizer.zero_grad()
        loss.backward()
        if train:
            optimizer.step()
        losses.append(loss.item())
        prefix = 'Train' if train else 'Test'
        for batch_id in range(pred.shape[0]):
            ssims.append(ssim_metric(pred[batch_id].unsqueeze(0), labels[batch_id].unsqueeze(0)).item())
        if (iter_num+1)%20==0:
            print(f"{prefix}:: Epoch: {epoch} | iter{iter_num}/{len(dataloader)} | Loss: {np.mean(losses)} | SSIM: {np.mean(ssims)} \n Logs : {logs}\n")
    return np.mean(ssims)
    
if __name__ == '__main__':
    model = ModelAPI(91,91)
    # model.load_state_dict(torch.load('/home/cavin/workspace/PetTauCVPR/weights/eca_ssim0.8980.pth'))
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
    criterion = nn.L1Loss()
    trainDataset = T1TauDataset(train=True)
    trainDataLoader = DataLoader(trainDataset,batch_size=2,shuffle=True)
    testDataset = T1TauDataset(train=False)
    testDataLoader = DataLoader(testDataset,batch_size=2)
    for i in range(5000):
        model.train()
        pred_one_epoch(i,model,trainDataLoader,optimizer,criterion,use_consistency=False)
        if (i+1)%20==0:
            model.eval()
            ssim = pred_one_epoch(i,model,testDataLoader,optimizer,criterion,train=False,use_consistency=False)
            print(ssim)
            torch.save(model.state_dict(),f'./weights/eca_ssim{ssim:.4f}.pth')
