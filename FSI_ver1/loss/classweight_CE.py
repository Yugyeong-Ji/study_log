import torch
import torch.nn as nn

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self,args ):
        
        """
        class_counts : list of number of classes 
        if there is two classes A & B, each of 100, and 10 samples
        class_counts = [100, 10]
        """
        
        super(WeightedCrossEntropyLoss, self).__init__()
        self.args = args
        total_samples = sum(self.args.class_counts)
        class_weights = [total_samples / count for count in self.args.class_counts]
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(args.device)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
    def forward(self, output, target):
        return self.criterion(output, target)