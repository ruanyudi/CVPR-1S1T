from dataset.T1AbetaDataset import T1ABetaDataset
from dataset.T1TauDataset import T1TauDataset
from dataclasses import dataclass


@dataclass
class Config:
    dataset = T1ABetaDataset
    use_consistency=False
