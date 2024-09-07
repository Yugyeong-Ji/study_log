import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self, args):
        super(MSELoss, self).__init__()
        self.args = args
        self.criterion = nn.MSELoss()
        
    def forward(self, output, target):
        return self.criterion(output, target)