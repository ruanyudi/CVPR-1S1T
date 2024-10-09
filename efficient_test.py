import torch
import torchvision
from torch import nn
import time
import numpy as np
from torchmetrics import StructuralSimilarityIndexMeasure
from torch.utils.data import DataLoader
from ModelAPI import ModelAPI
import matplotlib.pyplot as plt
device = 'mps'
ssim_metric=StructuralSimilarityIndexMeasure()
ssim_metric.to(device)

def pred_one_epoch(epoch,model:nn.Module,dataloader:torch.utils.data.DataLoader,optimizer:torch.optim,criterion:torch.nn,train=True):
    losses = []
    dataloader = dataloader
    ssims = []
    for iter_num, (data,labels) in enumerate(dataloader):
        logs = {}
        start_time = time.time()
        data,labels = data.to(device),labels.to(device)
        labels = data
        pred,instance_x,modality_x = model(data)
        loss = criterion(pred,labels)
        logs.update({'mse_loss':loss.item()})
        embed = lambda x : model.getEmbeddings(*x)
        logs.update({'cost_time':time.time()-start_time})
        optimizer.zero_grad()
        loss.backward()
        if train:
            optimizer.step()
        losses.append(loss.item())
        prefix = 'Train' if train else 'Test'
        for batch_id in range(pred.shape[0]):
            ssims.append(ssim_metric(pred[batch_id].unsqueeze(0), labels[batch_id].unsqueeze(0)).item())
        if (iter_num+1)%2==0:
            fig,axes= plt.subplots(1,2)
            axes[0].imshow(pred[0].detach().cpu().permute(1,2,0).numpy())
            axes[1].imshow(data[0].detach().cpu().permute(1,2,0).numpy())
            plt.savefig('cifar_efficient.png')
            torch.save(model.state_dict(),'cifar_efficient.pt')
            print(f"{prefix}:: Epoch: {epoch} | iter{iter_num}/{len(dataloader)} | Loss: {np.mean(losses)} | SSIM: {np.mean(ssims)} \n Logs : {logs}\n")
    return np.mean(ssims)

if __name__ == '__main__':
    model=ModelAPI(1,1)
    model.load_state_dict(torch.load('./cifar_efficient.pt'))
    model.to(device)
    trainDataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform=torchvision.transforms.ToTensor())
    testDataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=torchvision.transforms.ToTensor())
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=1, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=1, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = torch.nn.MSELoss()
    for epoch in range(1,100):
        model.train()
        start_time = time.time()
        ssim = pred_one_epoch(epoch,model,trainDataLoader,optimizer,criterion,train=True)
        end_time = time.time()
        print(f'Epoch {epoch} | Time: {end_time-start_time} | SSIM: {ssim}')
        if (epoch+1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                ssim = pred_one_epoch(epoch,model,testDataLoader,optimizer,criterion,train=False)

