import pandas as pd
import torch
import os 
from torch.utils.data import Dataset
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer
import numpy as np
import warnings


pd.set_option('future.no_silent_downcasting', True)
warnings.simplefilter(action='ignore', category=FutureWarning)

class CSVPairDataset(Dataset):
    def __init__(self, args, data_path: str):
        self.args = args
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.scaler = MaxAbsScaler()
        # self.scaler = RobustScaler()
        # self.scaler = MinMaxScaler()
        # Define the scaler you want to use
        # self.scaler = QuantileTransformer(output_distribution='normal')
        self.preprocess()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        input = self.features[idx].copy()  # Ensure we don't modify the original data
        label = self.labels[idx]

        # Apply augmentations here if in training mode
        if not self.args.test and not self.args.val and self.args.augmentation:
            input = self.apply_dynamic_augmentation(input)

        # Convert to tensor after augmentations
        input = torch.tensor(input, dtype=torch.float32).to(self.args.device)
        label = torch.tensor(label, dtype=torch.float32).to(self.args.device)
        
        # Test, extract the feature
        if self.args.test:
            ids = self.ids[idx] if hasattr(self, 'ids') else idx  # Use index as ID if 'ID' column is missing
            return input, ids, label
        
        # Train
        else:
            return input, label

    def preprocess(self):
        self.load_csv_file()

        # Extract features and labels
        if self.args.test:
            try:
                self.ids = self.data['Original_Index'].values
            except:
                pass
        
        try:
            self.labels = self.data['Fraud_Type'].values  # Assuming 'Fraud_Type' is the target column
        except:
            self.labels = np.zeros(len(self.data))
            print(f"No labels, probably test_set")
        
        # Drop unnecessary columns
        self.features = self.data.drop(columns=['ID', 'Fraud_Type', 'Original_Index'], errors='ignore').copy()
        print("Total number of samples:", len(self.features))
        print("Columns (or Input Size) used:", len(self.features.columns))
        
        # Label Encoding for 'Fraud_Type' to make labels numeric
        le_subclass = LabelEncoder()
        self.labels = le_subclass.fit_transform(self.labels)
        joblib.dump(le_subclass, os.path.join(self.args.save_path, 'label_encoder.pkl'))
    
        self.apply_transformations()

        # Print transformed labels
        for i, label in enumerate(le_subclass.classes_):
            print(f"Original Label: {label}, Transformed Number: {i}")
        
        if self.features.shape[1] != self.args.input_size:
            print(f"Resizing features from {self.features.shape[1]} to {self.args.input_size}")
            self.features = self.resize_features(self.features, self.args.input_size)

    def apply_transformations(self):
        date_columns = [
            'Customer_registration_datetime',
            'Account_creation_datetime',
            'Transaction_Datetime',
            'Last_atm_transaction_datetime',
            'Last_bank_branch_transaction_datetime',
            'Transaction_resumed_date'
        ]
        
        numerical_columns = []

        # Convert 'Time_difference' to seconds and add to numerical columns
        self.features['Time_difference_seconds'] = pd.to_timedelta(self.features['Time_difference']).dt.total_seconds()
        numerical_columns.append('Time_difference_seconds')

        # Convert the specified columns to datetime format and create differences
        for column in date_columns:
            self.features[column] = pd.to_datetime(self.features[column])
        
        # Create differences between the date columns
        for i in range(len(date_columns)):
            for j in range(i + 1, len(date_columns)):
                col_name = f'Diff_{date_columns[i]}_to_{date_columns[j]}'
                self.features[col_name] = (self.features[date_columns[j]] - self.features[date_columns[i]]).dt.days
                numerical_columns.append(col_name)

        # Drop the original date columns and the 'Time_difference' column
        self.features.drop(columns=['Time_difference'] + date_columns, axis=1, inplace=True)

        # Drop additional unnecessary columns
        self.features.drop(columns=[
            'Customer_personal_identifier',
            'Customer_identification_number',
            'Account_account_number',
            'Error_Code', 
            'Type_General_Automatic',
            'IP_Address',
            'MAC_Address',
            'Location',
            'Recipient_Account_Number',
            'Another_Person_Account',
            'First_time_iOS_by_vulnerable_user'
        ], axis=1, inplace=True)
        
        print(f"All columns: {self.features.columns.tolist()}\nNumber of columns: {self.features.shape[1]}")
        
        # Extend numerical columns list
        numerical_columns.extend([
            'Customer_Birthyear',
            'Account_initial_balance',
            'Account_balance',
            'Account_amount_daily_limit',
            'Account_remaining_amount_daily_limit_exceeded',
            'Account_one_month_max_amount',
            'Account_one_month_std_dev',
            'Account_dawn_one_month_max_amount',
            'Account_dawn_one_month_std_dev',
            'Transaction_Amount',
            'Distance',
            'Transaction_history_with_the_account',
        ])  # Replace with your actual numerical columns

        # All other columns are considered categorical
        categorical_columns = [col for col in self.features.columns if col not in numerical_columns]
        print(f"Categorical columns: {categorical_columns}")

        # Apply Target Encoding to categorical columns
        target_encoder = TargetEncoder(cols=categorical_columns, smoothing=0.3)
        self.features[categorical_columns] = target_encoder.fit_transform(self.features[categorical_columns], self.labels)
        
        # Save the encoder for future use
        joblib.dump(target_encoder, os.path.join(self.args.save_path, 'target_encoder.pkl'))
        
        # Normalize Numerical Variables
        self.features[numerical_columns] = self.scaler.fit_transform(self.features[numerical_columns])
        
        # Save the scaler for future use
        joblib.dump(self.scaler, os.path.join(self.args.save_path, 'scaler.pkl'))

        self.features = np.asarray(self.features.astype(np.float32))

    def resize_features(self, features, target_size):
        current_size = features.shape[1]
        if current_size > target_size:
            return features[:, :target_size]  # Truncate the features if larger
        elif current_size < target_size:
            padding = np.zeros((features.shape[0], target_size - current_size), dtype=np.float32)
            return np.hstack((features, padding))  # Pad the features with zeros if smaller
        
        return features

    def load_csv_file(self):
        self.data = pd.read_csv(self.data_path)
        self.data['Original_Index'] = self.data.index

        if not self.args.test:
            if self.args.use_all:
                self.synthetic_data = pd.read_csv(os.path.join(self.args.data_path, self.args.synthetic_path))
                self.synthetic_data['Original_Index'] = self.synthetic_data.index + 300000000
                self.data = pd.concat([self.data, self.synthetic_data], ignore_index=True)

        else:
            self.synthetic_data = pd.read_csv(os.path.join(self.args.data_path, self.args.synthetic_path))
            self.synthetic_data['Original_Index'] = self.synthetic_data.index + 300000000
            if self.args.use_all:
                self.data = pd.concat([self.data, self.synthetic_data], ignore_index=False)
                
    def apply_dynamic_augmentation(self, input):
        masking_prob = 0.10
        input_augmented = input.copy()
        mask = np.random.rand(input.shape[0]) < masking_prob
        input_augmented[mask] = 0.0
        return input_augmented