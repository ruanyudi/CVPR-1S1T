from dataset import T1TauDataset,T1AbetaDataset
from dataclasses import dataclass
@dataclass
class Config:
    dataset=T1TauDataset