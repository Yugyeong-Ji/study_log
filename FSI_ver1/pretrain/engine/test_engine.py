import torch
from tqdm import tqdm
from collections import defaultdict
from metrics.utils import save_submission_csv
import numpy as np
import os
import pandas as pd


def test_engine(args, model, dataloader):
    
    model.eval()
    
    if args.task == 'mean':
        features_by_label = defaultdict(list)
        
        # Compute mean features using the dataloader
        with torch.no_grad():
            for features, ids, labels in tqdm(dataloader, desc="Calculating Mean Features"):
                outputs = model(features)  # Outputs shape: (batch_size, 128)

                # Accumulate features by label
                for i, label in enumerate(labels):
                    features_by_label[label.item()].append(outputs[i].cpu().numpy())

        # Compute mean features for each label
        mean_features_by_label = {}
        for label, features in features_by_label.items():
            mean_features_by_label[label] = np.mean(features, axis=0)  # Mean across the batch dimension


        # Save the mean features by label
        save_mean_features(args, mean_features_by_label)
        
        print(f"saved the mean features")
    
    
    elif args.task == 'extract':
        all_ids = []
        all_features = None
        all_labels = []
        
        # Extract all features using the dataloader
        with torch.no_grad():
            for features, ids, labels in tqdm(dataloader, desc="Extracting All Features"):
                outputs = model(features)  # Outputs shape: (batch_size, 128)

                # Concatenate outputs and labels across batches
                all_features = torch.cat([all_features, outputs.cpu()]) if all_features is not None else outputs.cpu()
                all_labels.extend(labels.cpu().tolist())  # Store the labels as a list of integers
                all_ids.extend(ids)

        # Save all the extracted features and corresponding labels/IDs
        save_all_features(args, all_features, all_labels, all_ids)

        print(f"saved all_features")


def save_all_features(args, all_features, all_labels, all_ids):
    # Convert features to a numpy array if it's a tensor
    all_features_np = all_features.numpy() if torch.is_tensor(all_features) else all_features

    # Combine IDs, features, and labels into a single DataFrame
    labels_and_ids = np.column_stack((all_ids, all_features_np, all_labels))  # Combine IDs, features, and labels

    # Create a DataFrame with appropriate column names
    num_features = all_features_np.shape[1]
    feature_columns = [f'Feature_{i}' for i in range(num_features)]
    columns = ['ID'] + feature_columns + ['Fraud_Type']
    
    labels_df = pd.DataFrame(labels_and_ids, columns=columns)

    # Save the DataFrame to a CSV file
    labels_df.to_csv(os.path.join(args.feature_path, 'all_features.csv'), index=False)



def save_mean_features(args, mean_features_by_label):
    # Convert mean features by label to a DataFrame
    mean_features_df = pd.DataFrame.from_dict(mean_features_by_label, orient='index')
    mean_features_df.index.name = 'Fraud_Type'
    mean_features_df.columns = [f'Feature_{i}' for i in range(mean_features_df.shape[1])]

    # Save mean features to a CSV file
    mean_features_df.to_csv(os.path.join(args.feature_path, 'mean_features_by_label.csv'))