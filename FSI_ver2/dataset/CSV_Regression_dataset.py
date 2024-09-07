import pandas as pd
import torch
import os 
from torch.utils.data import Dataset
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


import numpy as np

class CSVRegressionDataset(Dataset):
    
    def __init__(self, args, data_path: str):
        self.args = args
        self.data_path = data_path
        self.preprocess()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.args.test:
            input = torch.tensor(self.features[idx], dtype=torch.float32).to(self.args.device)
            ids = self.ids[idx]
            return input, ids
        else:
            input = torch.tensor(self.features[idx], dtype=torch.float32).to(self.args.device)
            label = input.clone()  # For autoencoder, input and label are the same
            return input, label

    def preprocess(self):
        self.load_csv_file()

        if self.args.test:
            # Ensure the same transformation as training for the test set
            self.ids = self.data['ID'].values
            self.features = self.data.drop(columns=['ID']).copy()
            
            # Sample 1000 instances for each label in the 63rd column
            def sample_group(group):
                return group.sample(n=1000, random_state=42, replace=False)  # Use replace=True to handle cases with less than 1000 samples

            sampled_data = self.synthetic_data.groupby(self.synthetic_data.columns[62]).apply(sample_group).reset_index(drop=True)
            # Drop the first column
            sampled_data = sampled_data.drop(sampled_data.columns[0], axis=1)
            
            sampled_data.to_csv(os.path.join(self.args.save_path, 'syn_submission.csv'), index=False)
            self.apply_transformations()
            
        else:
            # Extract features and labels for training
            self.features = self.data.drop(columns=['ID', 'Fraud_Type']).copy()
            self.labels = self.features 
            self.apply_transformations()

            
        if self.features.shape[1] != self.args.input_size:
            print(f"Resizing features from {self.features.shape[1]} to {self.args.input_size}")
            self.features = self.resize_features(self.features, self.args.input_size)

    def apply_transformations(self):
        # Encoding categorical variables in features
        categorical_columns = self.features.select_dtypes(include=['object', 'category']).columns
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        # Apply ordinal encoding to categorical columns
        self.features[categorical_columns] = ordinal_encoder.fit_transform(self.features[categorical_columns])
        
        # Save the encoder for future use
        joblib.dump(ordinal_encoder, os.path.join(self.args.save_path, 'ordinal_encoder.pkl'))

        # Normalize the features
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)
        
        # Save the scaler for future use
        joblib.dump(scaler, os.path.join(self.args.save_path, 'scaler.pkl'))

        # Save a sample of the preprocessed features for inspection
        pd.DataFrame(self.features).head().to_csv(os.path.join(self.args.save_path, 'feature_preprocessed_example.csv'), index=False)

        # Convert to numpy array
        self.features = self.features.astype(np.float32)

    def resize_features(self, features, target_size):
        current_size = features.shape[1]
        if current_size > target_size:
            # If the current size is larger, truncate the features
            return features[:, :target_size]
        elif current_size < target_size:
            # If the current size is smaller, pad the features with zeros
            padding = np.zeros((features.shape[0], target_size - current_size), dtype=np.float32)
            return np.hstack((features, padding))
        return features

    def load_csv_file(self):
        # Load the main dataset
        self.data = pd.read_csv(self.data_path)
        
        # If use_all flag is set, load and concatenate synthetic data
        if self.args.use_all:
            self.synthetic_data = pd.read_csv(os.path.join(self.args.data_path, self.args.synthetic_path))
            
            # Concatenate the main data with synthetic data
            self.data = pd.concat([self.data, self.synthetic_data], ignore_index=True)
                
            # Print basic information about the dataset
            # print("Columns:", self.data.columns.tolist())
            print("Total number of Rows:", len(self.data))
            print("Total number of Columns:", len(self.data.columns))
            
        if self.args.test:
            self.synthetic_data = pd.read_csv(os.path.join(self.args.data_path, self.args.synthetic_path))

