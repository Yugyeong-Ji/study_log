import pandas as pd
import torch
import os 
from torch.utils.data import Dataset
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer
from sklearn.model_selection import train_test_split

import numpy as np

class CSVBinaryDataset(Dataset):
    
    def __init__(self, args, data_path: str):
        self.args = args
        self.data_path = data_path
        self.preprocess()


    def __len__(self):
        return len(self.features)


    def __getitem__(self, idx):
        
        input = self.features[idx].copy()  # Ensure we don't modify the original data
        if not self.args.test:
            label = self.labels[idx]
            
            
        # Apply augmentations here if in training mode
        if not self.args.test and not self.args.val and self.args.augmentation:

            input = self.apply_dynamic_augmentation(input)

        # Convert to tensor after augmentations
        input = torch.tensor(input, dtype=torch.float32).to(self.args.device)

        
        if self.args.test:
            ids = self.ids[idx]
            return input, ids
        
        else :
            label = torch.tensor(label, dtype=torch.long).to(self.args.device)
            
            return input, label


    def preprocess(self):
        
        self.load_csv_file()

        if self.args.test:
            # Ensure the same transformation as training for the test set
            self.ids = self.data['ID'].values
            self.features = self.data.drop(columns=['ID']).copy()
            
            self.apply_transformations()
            
        else:
            # Extract features and labels for training
            self.labels = self.data['Fraud_Type'].values  # Assuming 'Fraud_Type' is the target column
            
            try:
                self.features = self.data.drop(columns=['ID', 'Fraud_Type']).copy()
            except:
                self.features = self.data.drop(columns=['Fraud_Type']).copy()
            self.apply_transformations()

            # Label Encoding for 'Fraud_Type'
            le_subclass = LabelEncoder()
            self.labels = le_subclass.fit_transform(self.labels)
            joblib.dump(le_subclass, os.path.join(self.args.save_path, 'label_encoder.pkl'))

            # Print transformed labels
            for i, label in enumerate(le_subclass.classes_):
                print(f"Original Label: {label}, Transformed Number: {i}")

            if not self.args.val:
                self.calculate_class_counts()
            
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
        #scaler = StandardScaler()
        # scaler = MaxAbsScaler()
        #scaler = RobustScaler()
        #scaler = MinMaxScaler()
        scaler = QuantileTransformer(output_distribution='normal')
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
        if self.args.use_all and not self.args.val:
            self.synthetic_data = pd.read_csv(os.path.join(self.args.data_path, self.args.synthetic_path))
            
            # Concatenate the main data with synthetic data
            self.data = pd.concat([self.data, self.synthetic_data], ignore_index=True)
                
            # Print basic information about the dataset
            # print("Columns:", self.data.columns.tolist())
            print("Total number of Rows:", len(self.data))
            print("Total number of Columns:", len(self.data.columns))
            
        if self.args.test:
            self.synthetic_data = pd.read_csv(os.path.join(self.args.data_path,self.args.synthetic_path))
        

    def calculate_class_counts(self):
        self.class_counts = np.bincount(self.labels)
        self.args.class_counts = self.class_counts



    def apply_dynamic_augmentation(self, input):
        masking_prob = 0.25
        input_augmented = input.copy()

        # Generate a mask for the input array
        mask = np.random.rand(input.shape[0]) < masking_prob

        # Set the masked elements to zero (or another neutral value)
        input_augmented[mask] = 0.0

        return input_augmented