import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self,args):
        super(CrossEntropyLoss, self).__init__()
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, output, target):
        return self.criterion(output, target)