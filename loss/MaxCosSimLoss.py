import torch
import torch.nn as nn
import torch.nn.functional as F


class MaximizeCosineDistanceLoss(nn.Module):
    def __init__(self):
        """
        Initializes the loss function that maximizes the cosine distance between two vectors.
        """
        super(MaximizeCosineDistanceLoss, self).__init__()

    def forward(self, z_i, z_j):
        """
        Compute the loss that maximizes the cosine distance between two input vectors.

        Parameters:
        - z_i: First input tensor of shape (B, N).
        - z_j: Second input tensor of shape (B, N).

        Returns:
        - Loss value (always positive).
        """
        # Normalize the vectors to unit length
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Compute cosine similarity
        cosine_similarity = F.cosine_similarity(z_i, z_j, dim=1)
        # print(cosine_similarity.shape)

        # Loss is 1 - cosine_similarity to ensure positive loss
        loss = torch.mean(1 - cosine_similarity)  # Loss is in range [0, 2]

        return loss

# Example usage:
# z_i, z_j are (B, N) shape vectors
# loss_fn = MaximizeCosineDistanceLoss()
# loss = loss_fn(z_i, z_j)
if __name__ == '__main__':
    x = torch.randn(4, 128)
    y = torch.randn(4, 128)
    loss = MaximizeCosineDistanceLoss()(x, y)
    print(loss)