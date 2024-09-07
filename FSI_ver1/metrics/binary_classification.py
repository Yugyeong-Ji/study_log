import torch
from sklearn.metrics import accuracy_score


"""For imbalanced dataset, we need to achieve score for positive labels & Negative labels separately """
def evaluate(args, labels, predictions):
    # Apply threshold
    # print(f"labels, predictions shape : {labels.shape, predictions.shape}")
    thresholded_predictions = (predictions > args.threshold).float()
    
    # Convert to numpy arrays for sklearn metrics
    labels = labels.numpy()
    thresholded_predictions = thresholded_predictions.numpy()
    
    # print(labels)
    # print(thresholded_predictions)
    # Calculate metrics
    accuracy = accuracy_score(labels, thresholded_predictions)

    return  {
        "accuracy": accuracy
    }
