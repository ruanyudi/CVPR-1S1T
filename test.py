from time import sleep
from dataset.T1TauDataset import T1TauDataset
from ModelAPI import ModelAPI
import torch
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure
import numpy as np
import matplotlib.pyplot as plt
import nibabel
from dataset.BrainPostProcess import BrainPostProcess
from tqdm import tqdm
SSIM = StructuralSimilarityIndexMeasure()

DEVICE='cpu'
CHANNEL_SIZE = 91
SSIM.to(DEVICE)
brainPostProcess = BrainPostProcess()
brainPostProcess.to(DEVICE)

def display_imgs(img0,img1,img2,img3,img4):
    fig,axes = plt.subplots(1,5,figsize=(15,5))
    images = [img0,img1,img2,img3,img4]
    titles=['input','gt','pred','instance','modality']
    for i,ax in enumerate(axes):
        ax.imshow(images[i],cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')
    plt.savefig('./test_visual.png')
    plt.cla()


if __name__ == '__main__':
    model = ModelAPI(CHANNEL_SIZE,CHANNEL_SIZE)
    model.load_state_dict(torch.load('./weights/eca_ssim0.8158.pth'))
    model.to(DEVICE)
    model.eval()
    dataset = T1TauDataset(train=False)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
    ssims = []
    dataloader = tqdm(dataloader)
    for i,(imgs,labels) in enumerate(dataloader):
        SelectedChannel = np.random.randint(0,CHANNEL_SIZE)
        imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
        pred,instance,modality = model(imgs)
        pred = brainPostProcess(pred)
        for batch_id in range(pred.shape[0]):
            ssims.append(SSIM(pred[batch_id].unsqueeze(0),labels[batch_id].unsqueeze(0)).item())
        pred=pred.squeeze().cpu().detach().numpy()
        nii_image = nibabel.Nifti1Image(pred.transpose(1,2,0),affine=np.eye(4))
        nibabel.save(nii_image,f'./results/{i}.nii')
        imgs=imgs.squeeze().cpu().detach().numpy()
        labels=labels.squeeze().cpu().detach().numpy()
        instance=instance.squeeze().cpu().detach().numpy()
        modality=modality.squeeze().cpu().detach().numpy()
        pred = pred[SelectedChannel]
        imgs = imgs[SelectedChannel]
        labels = labels[SelectedChannel]
        instance=instance[SelectedChannel]
        modality=modality[SelectedChannel]
        display_imgs(imgs,labels,pred,instance,modality)
        # print(np.mean(s_ssim))
        # print(SSIM(pred,labels))
    print(np.mean(ssims))
