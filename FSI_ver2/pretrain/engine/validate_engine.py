import torch
from tqdm import tqdm

def validate_engine(args, model, dataloader, criterion):
    
    model.eval()
    total_loss = 0.0
    all_labels = None
    all_predictions = None

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validating"):
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()


    # Calculate metrics


    average_loss = total_loss / len(dataloader)
    
    return average_loss, all_labels, all_predictions

