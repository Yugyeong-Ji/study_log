import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module): 
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()
        self.temperature = args.temperature

    def forward(self, tokens, labels):
        """
        Computes the contrastive loss for multi-class labels using the InfoNCE approach.
        
        Parameters:
        - tokens: Tensor of shape (batch_size, embedding_dim), the input token embeddings.
        - labels: Tensor of shape (batch_size,), the class labels for each token.
        
        Returns:
        - loss: The computed contrastive loss.
        """
        batch_size = tokens.size(0)
        
        # Compute cosine similarity matrix between all pairs
        cosine_similarity = F.cosine_similarity(tokens.unsqueeze(1), tokens.unsqueeze(0), dim=-1)  # Shape: (batch_size, batch_size)

        # Scale similarities by temperature
        logits = cosine_similarity / self.temperature
        
        # Create pairwise labels: 1 if same class, 0 if different class
        labels = labels.unsqueeze(0)  # Shape: (1, batch_size)
        pairwise_labels = (labels == labels.T).float()  # Shape: (batch_size, batch_size)
        
        # Mask out self-comparisons by ignoring diagonal elements in both logits and labels
        mask = torch.eye(batch_size, dtype=torch.bool).to(tokens.device)  # Shape: (batch_size, batch_size)
        logits = logits.masked_fill(mask, -1e9)  # Set self-similarity logits to a very small number
        pairwise_labels = pairwise_labels.masked_fill(mask, 0)  # Ignore self-comparisons in labels
        
        # Apply log-softmax to logits for numerical stability
        log_probs = F.log_softmax(logits, dim=1)
        
        # Compute the contrastive loss
        # We use pairwise_labels as weights, focusing on positive pairs
        loss = - (pairwise_labels * log_probs).sum() / pairwise_labels.sum()
        
        return loss