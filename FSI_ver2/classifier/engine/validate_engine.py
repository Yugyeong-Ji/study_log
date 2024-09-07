import torch
from tqdm import tqdm

from metrics import load_metric

from metrics.utils import save_labels_and_predictions

def validate_engine(args, model, dataloader, criterion):
    
    model.eval()
    total_loss = 0.0
    all_labels = None
    all_predictions = None
    evaluate = load_metric(args)
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validating"):
            outputs = model(features)
            
            try:
                ouputs = outputs.squeeze()
            except:
                pass
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # Convert outputs to binary predictions (0 or 1)
            all_labels = torch.cat([all_labels, labels.cpu()]) if all_labels is not None else labels.cpu()
            all_predictions = torch.cat([all_predictions, outputs.cpu()]) if all_predictions is not None else outputs.cpu()
   
    # Calculate metrics
    score = evaluate(args, all_labels, all_predictions)
    average_loss = total_loss / len(dataloader)
    
    return average_loss, score, all_labels, all_predictions

