import torch
import torch.nn as nn
import torch.nn.functional as F


class EuclideanContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(EuclideanContrastiveLoss, self).__init__()
        self.temperature = args.temperature

    def forward(self, tokens, labels):
        """
        Computes the contrastive loss for multi-class labels using Euclidean distance.
        
        Parameters:
        - tokens: Tensor of shape (batch_size, embedding_dim), the input token embeddings.
        - labels: Tensor of shape (batch_size,), the class labels for each token.
        
        Returns:
        - loss: The computed contrastive loss.
        """
        batch_size = tokens.size(0)
        tokens = F.normalize(tokens, p=2, dim=1)
        # Compute pairwise Euclidean distances between all token embeddings
        diffs = tokens.unsqueeze(1) - tokens.unsqueeze(0)  # Shape: (batch_size, batch_size, embedding_dim)
        distances = torch.norm(diffs, dim=-1)  # Shape: (batch_size, batch_size)

        # Scale distances by temperature (optional)
        logits = -distances / self.temperature  # Shape: (batch_size, batch_size)
        
        # Create pairwise labels: 1 if same class, 0 if different class
        labels = labels.unsqueeze(0)  # Shape: (1, batch_size)
        pairwise_labels = (labels == labels.T).float()  # Shape: (batch_size, batch_size)
        
        # Mask out self-comparisons
        mask = torch.eye(batch_size, dtype=torch.bool).to(tokens.device)  # Shape: (batch_size, batch_size)
        logits = logits.masked_fill(mask, -1e9)  # Set self-distance logits to a very small number
        pairwise_labels = pairwise_labels.masked_fill(mask, 0)  # Ignore self-comparisons in labels
        
        # Apply log-softmax to logits for numerical stability
        log_probs = F.log_softmax(logits, dim=1)
        
        # Compute the contrastive loss
        loss = - (pairwise_labels * log_probs).sum() / pairwise_labels.sum()
        
        return loss