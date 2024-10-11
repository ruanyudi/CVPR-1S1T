import nibabel
import torch
import os 
from torch.utils.data import Dataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt

T1_path = './data/T1_PEt/T1'
Tau_path = './data/T1_PEt/TauPET'
train_split=0.8


class T1TauDataset(Dataset):
    def __init__(self,train=True) -> None:
        super().__init__()
        self.T1_filepaths = os.listdir(T1_path)
        self.Tau_filepaths = os.listdir(Tau_path)
        assert len(self.T1_filepaths)==len(self.Tau_filepaths)
        T1_id2file = {}
        Tau_id2file = {}
        for file in self.T1_filepaths:
            id = file.split('_')[2].split('-')[0]
            T1_id2file.update({id:os.path.join(T1_path,file)})
        for file in self.Tau_filepaths:
            id = file.split('_')[3].split('-')[0]
            Tau_id2file.update({id:os.path.join(Tau_path,file)})
        self.index2id = list(T1_id2file.keys())
        # self.index2id = self.index2id[:10]
        self.T1_id2file = T1_id2file
        self.Tau_id2file = Tau_id2file

        self.train_split = train_split
        self.len = len(self.index2id)
        if train:
            self.index2id = self.index2id[:int(self.train_split*self.len)]
        else:
            self.index2id = self.index2id[int(self.train_split*self.len):]
        self.len = len(self.index2id)

    def __len__(self):
        return self.len

    def get_nidata(self,filepath:str):
        ni_data = nibabel.load(filepath)
        ni_data = ni_data.get_fdata()
        return ni_data
    

    def __getitem__(self, index):
        id = self.index2id[index]
        T1_filepath = self.T1_id2file[id]
        Tau_filepath = self.Tau_id2file[id]
        T1_data = self.get_nidata(T1_filepath)
        Tau_data=self.get_nidata(Tau_filepath)
        T1_data = torch.tensor(T1_data,dtype=torch.float32).permute(2,0,1)
        Tau_data=torch.tensor(Tau_data,dtype=torch.float32).permute(2,0,1)
        return T1_data,Tau_data

if __name__ == '__main__':
    dataset = T1TauDataset()
    dataloader = DataLoader(dataset,batch_size=1)
    for i,(data,label) in enumerate(dataloader):
        print(dataset.Tau_filepaths[dataset.index2id[i]])
        print(dataset.T1_id2file[dataset.index2id[i]])
        print(data.shape)
        print(label.shape)
        exit(0)