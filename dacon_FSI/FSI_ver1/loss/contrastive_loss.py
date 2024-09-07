import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()
        self.temperature = args.temperature

    def forward(self, batch_outputs, labels):
        batch_size = batch_outputs.size(0)
        
        # Normalize the batch outputs (L2 normalization)
        batch_outputs = F.normalize(batch_outputs, p=2, dim=1)
        
        # Compute the cosine similarity matrix
        similarity_matrix = torch.matmul(batch_outputs, batch_outputs.T) / self.temperature
        
        # Create a binary mask where mask[i, j] = 1 if labels[i] == labels[j], else 0
        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.T).float().to(batch_outputs.device)
        
        # Mask out self-comparisons (diagonal elements) in the similarity matrix
        self_mask = torch.eye(batch_size, dtype=torch.bool).to(batch_outputs.device)
        similarity_matrix = similarity_matrix.masked_fill(self_mask, -1e9)
        
        # Calculate log-softmax over similarity matrix
        log_probs = F.log_softmax(similarity_matrix, dim=1)
        
        # Mask out the diagonal elements from the mask
        mask = mask.masked_fill(self_mask, 0)
        
        # Calculate the positive loss for pairs with the same label
        positive_loss = - (mask * log_probs).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # Exclude diagonal elements in negative loss calculation
        negative_mask = (1 - self_mask.float()) * (1 - mask)
        negative_loss = - (negative_mask * log_probs).sum(dim=1) / (negative_mask.sum(dim=1) + 1e-8)
        
        # Combine positive and negative loss and average over the batch
        loss = (positive_loss + negative_loss).mean()

        return loss