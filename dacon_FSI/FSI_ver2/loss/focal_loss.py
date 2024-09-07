import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, args):
        super(FocalLoss, self).__init__()
        self.gamma = args.gamma
        self.alpha = None

    def forward(self, output, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(output, target)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, target)
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()