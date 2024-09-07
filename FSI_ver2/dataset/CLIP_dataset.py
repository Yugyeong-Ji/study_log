import pandas as pd
import torch
import os 
from torch.utils.data import Dataset
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler


import numpy as np

class CSVPairDataset(Dataset):
    def __init__(self, args, data_path: str):
        self.args = args
        self.data_path = data_path
        self.preprocess()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        input = torch.tensor(self.features[idx], dtype=torch.float32).to(self.args.device)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).to(self.args.device)
        # Test, extract the feature
        if self.args.test:
            # try :
            ids = self.ids[idx]
            # except:
            #     ids = idx  # Use index as ID if 'ID' column is missing

            return input, ids, label
        
        # train
        else :
            
            return input, label


    def preprocess(self):
        self.load_csv_file()

        # Extract features and labels
        if self.args.test:
            try :
                self.ids = self.data['Original_Index'].values
                
            except:
                pass

        self.labels = self.data['Fraud_Type'].values  # Assuming 'Fraud_Type' is the target column
        self.features = self.data.drop(columns=['ID', 'Fraud_Type', 'Original_Index'], errors='ignore').copy()
        # Print basic information about the dataset
        print("Total number of samples:", len(self.features))
        print("Columns ( or Input Size ) used:", len(self.features.columns))
        self.apply_transformations()

        # Label Encoding for 'Fraud_Type'
        le_subclass = LabelEncoder()
        self.labels = le_subclass.fit_transform(self.labels)
        joblib.dump(le_subclass, os.path.join(self.args.save_path, 'label_encoder.pkl'))

        # Print transformed labels
        for i, label in enumerate(le_subclass.classes_):
            print(f"Original Label: {label}, Transformed Number: {i}")

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
        self.data['Original_Index'] = self.data.index

        # train
        if not self.args.test:
            # If use_all flag is set, load and concatenate synthetic data
            
            if self.args.use_all:
                self.synthetic_data = pd.read_csv(os.path.join(self.args.data_path, self.args.synthetic_path))
                self.synthetic_data['Original_Index'] = self.synthetic_data.index + 300000000
                # Concatenate the main data with synthetic data
                self.data = pd.concat([self.data, self.synthetic_data], ignore_index=True)
                    

        # Test
        else:
            
            # Purpose : 1. extract top 1000 2. Feature extraction ( optional when self.args.use_all)
            self.synthetic_data = pd.read_csv(os.path.join(self.args.data_path, self.args.synthetic_path))
            
            self.synthetic_data['Original_Index'] = self.synthetic_data.index + 300000000
            if self.args.use_all:

                # Concatenate the main data with synthetic data
                self.data = pd.concat([self.data, self.synthetic_data], ignore_index=False)
                
                
