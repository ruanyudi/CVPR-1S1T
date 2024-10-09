import torch
from torch import nn
import nibabel
import numpy as np
class BrainPostProcess(nn.Module):
    def __init__(self):
        super(BrainPostProcess, self).__init__()
        brain_mask = nibabel.load('./brainmask.nii')
        brain_mask = brain_mask.get_fdata()
        brain_mask = np.transpose(brain_mask, (2, 0, 1))
        self.brain_mask = torch.from_numpy(brain_mask)
        self.brain_mask = self.brain_mask.to(dtype=torch.float64)
        self.brain_mask = torch.nn.Parameter(self.brain_mask,requires_grad=False)
    def forward(self, input):
        output = input*self.brain_mask
        return output


if __name__ == '__main__':
    brainPostProcess = BrainPostProcess()
    dummy_input = torch.randn(1, 91 , 91, 109)
    output = brainPostProcess(dummy_input)
    output = output.squeeze().numpy().transpose(1,2,0)
    output = nibabel.Nifti1Image(output,affine=np.eye(4))
    nibabel.save(output,'dummy_masked.nii')