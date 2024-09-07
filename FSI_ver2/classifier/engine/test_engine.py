import torch
from tqdm import tqdm

from metrics.multiclass_classification import evaluate

from metrics.utils import save_submission_csv

def test_engine(args, model, dataloader):
    
    model.eval()
    all_ids = []
    all_predictions = None
    
    with torch.no_grad():
        for features, ids in tqdm(dataloader, desc="Testing"):
            outputs = model(features).squeeze()
            # Convert outputs to binary predictions (0 or 1)
            all_predictions = torch.cat([all_predictions, outputs.cpu()]) if all_predictions is not None else outputs.cpu()
            all_ids.extend(ids)
        
    # Save submission file
    save_submission_csv(args, all_ids, all_predictions)
    
    return   all_predictions

