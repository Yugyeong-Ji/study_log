import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(args, labels, predictions):
    # Convert logits to predicted class indices
    
    predictions = torch.softmax(predictions,dim=1)
    predicted_classes = torch.argmax(predictions, dim=1)
    
    print(predicted_classes.shape)
    # Convert to numpy arrays for sklearn metrics
    labels = labels.numpy()
    predicted_classes = predicted_classes.numpy()
    
    # Calculate overall metrics
    accuracy = accuracy_score(labels, predicted_classes)
    precision = precision_score(labels, predicted_classes, average='weighted', zero_division=0)
    recall = recall_score(labels, predicted_classes, average='weighted')
    f1 = f1_score(labels, predicted_classes, average='weighted')

    # Calculate per-class metrics
    per_class_precision = precision_score(labels, predicted_classes, average=None, zero_division=0)
    per_class_recall = recall_score(labels, predicted_classes, average=None)
    per_class_f1 = f1_score(labels, predicted_classes, average=None)

    # Collect the metrics in a dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1
    }

    return metrics