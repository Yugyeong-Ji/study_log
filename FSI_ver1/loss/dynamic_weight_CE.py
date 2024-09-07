import torch
import torch.nn as nn

class DynamicWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, args):
        """
        class_counts : list of number of classes 
        if there are two classes A & B, each of 100, and 10 samples
        class_counts = [100, 10]
        """
        super(DynamicWeightedCrossEntropyLoss, self).__init__()
        self.args = args
        total_samples = sum(self.args.class_counts)
        
        # Compute class weights based on inverse of class frequencies
        class_weights = [total_samples / count for count in self.args.class_counts]
        
        # Normalize the class weights to sum up to 1
        weight_sum = sum(class_weights)
        class_weights = [w / weight_sum for w in class_weights]
        print(f"total_sample : {total_samples}")
        print(f"class_counts : {self.args.class_counts}")
        print(f"class_weights : {class_weights}")
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(args.device)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
        
        
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, output, target):
        # Calculate the initial loss without reduction
        loss = self.criterion(output, target)
        
        # Calculate the confidence scores (softmax probabilities)
        softmax_probs = torch.softmax(output, dim=1)
        
        # Get the predicted class and confidence
        pred_confidences, pred_classes = torch.max(softmax_probs, dim=1)
        
        # Create a mask for true positives
        true_positive_mask = (pred_classes == target).float()
        
        # Create a mask for false negatives where the confidence is significantly lower than the max confidence
        false_negative_mask = (pred_classes != target).float() * (pred_confidences < (self.args.threshold * softmax_probs.gather(1, target.view(-1, 1)).squeeze(1))).float()
        
        # Apply dynamic weights
        dynamic_weights = 0.5 * true_positive_mask + 0.5 * false_negative_mask + (1 - true_positive_mask) * (1 - false_negative_mask)
        
        # Apply the dynamic weights to the loss
        weighted_loss = loss * dynamic_weights
        
        # Reduce the loss (mean reduction)
        return weighted_loss.mean()