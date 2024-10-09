import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLossBatch(nn.Module):
    def __init__(self, temperature=0.5):
        """
        Initializes the NT-Xent (InfoNCE) Loss function for contrastive learning within a batch.

        Parameters:
        - temperature: Temperature scaling parameter for softmax.
        """
        super(NTXentLossBatch, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Compute the NT-Xent loss within a batch.

        Parameters:
        - z_i: Embeddings of first set of samples (shape: [B, N]).
        - z_j: Embeddings of second set of samples (shape: [B, N]).

        Returns:
        - Loss value.
        """
        # Normalize the embeddings to project them on the unit hypersphere
        z_i = F.normalize(z_i, dim=1)  # Shape: (B, N)
        z_j = F.normalize(z_j, dim=1)  # Shape: (B, N)

        # Concatenate z_i and z_j to create a batch of 2B samples
        z = torch.cat([z_i, z_j], dim=0)  # Shape: (2B, N)

        # Compute pairwise cosine similarities between all samples
        similarity_matrix = torch.mm(z, z.T)  # Shape: (2B, 2B)
        similarity_matrix /= self.temperature  # Apply temperature scaling

        # Create a mask to avoid comparing the same sample with itself
        batch_size = z_i.shape[0]
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix.masked_fill_(mask, float('-inf'))  # Set diagonal elements to -inf to ignore self-similarity

        # Positive pairs: for each sample i, its positive is at position i + batch_size
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0).to(z.device)
        # Compute cross entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss
