import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import warnings
from sklearn.utils import resample

# Ignore specific warnings
warnings.filterwarnings('ignore', message="There is an existing primary key 'ID'. This key will be removed.")
warnings.filterwarnings('ignore', message="We strongly recommend saving the metadata using 'save_to_json' for replicability in future SDV versions.")

def handle_outliers(series, n_std=3):
    mean = series.mean()
    std = series.std()
    z_scores = np.abs(stats.zscore(series))
    return series.mask(z_scores > n_std, mean)

def preprocess_data(train, fraud_type, n_sample):
    subset = train[train["Fraud_Type"] == fraud_type]

    if len(subset) < n_sample:
        subset = resample(subset, replace=True, n_samples=n_sample, random_state=42)
    else:
        subset = subset.sample(n=n_sample, random_state=42)

    # Convert and handle outliers
    subset['Time_difference_seconds'] = pd.to_timedelta(subset['Time_difference']).dt.total_seconds()
    subset['Time_difference_seconds'] = handle_outliers(subset['Time_difference_seconds'])
    subset = subset.drop('Time_difference', axis=1)
    
    return subset

def generate_synthetic_data(subset, metadata, n_cls_per_gen):
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=800,  # Set to 1000 for more training
        generator_lr=1e-4,  # Learning rate for the generator
        discriminator_lr=1e-4,  # Learning rate for the discriminator
        generator_dim=(256, 256, 128),  # Adding an extra layer to the generator
        discriminator_dim=(256, 256, 128),  # Adding an extra layer to the discriminator
        batch_size=500,  # Larger batch size for stability
        discriminator_steps=5,  # More updates to discriminator per generator step
        pac=5,  # Smaller groups in PAC
        log_frequency=True,  # Use log frequency for categorical sampling
        #verbose=True  # Enable verbose mode to monitor progress
    )
    
    synthesizer.fit(subset)
    synthetic_subset = synthesizer.sample(num_rows=n_cls_per_gen)
    
    return synthetic_subset


def postprocess_and_concat(synthetic_subset, train, all_synthetic_data, all_synthetic_data_submission):
    synthetic_subset['Time_difference'] = pd.to_timedelta(synthetic_subset['Time_difference_seconds'], unit='s')
    synthetic_subset = synthetic_subset.drop('Time_difference_seconds', axis=1)

    # Concatenate to all_synthetic_data 
    all_synthetic_data_submission = pd.concat([all_synthetic_data_submission, synthetic_subset], ignore_index=True)
    
    # Reorder columns to match the original train DataFrame
    """ㅈㄴ 이상한게, 이렇게 train이랑 column을 맞추면 오히려 문제가 생겨서 제출이 안됨;;;"""
    synthetic_subset = synthetic_subset[train.columns]

    # Concatenate to all_synthetic_data_submission
    all_synthetic_data = pd.concat([all_synthetic_data, synthetic_subset], ignore_index=True)
    
    return all_synthetic_data, all_synthetic_data_submission


def main_process(train, fraud_types, n_cls_per_gen, n_sample):
    all_synthetic_data = pd.DataFrame()
    all_synthetic_data_submission = pd.DataFrame()

    for fraud_type in tqdm(fraud_types):
        subset = preprocess_data(train, fraud_type, n_sample)

        # Create and update metadata for each fraud type subset
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(subset)
        metadata.set_primary_key(None)
        
        # Ensure the data types are set correctly
        column_sdtypes = {
            'Account_initial_balance': 'numerical',
            'Account_balance': 'numerical',
            'Customer_identification_number': 'categorical',  
            'Customer_personal_identifier': 'categorical',
            'Account_account_number': 'categorical',
            'IP_Address': 'ipv4_address',  
            'Location': 'categorical',
            'Recipient_Account_Number': 'categorical',
            'Fraud_Type': 'categorical',
            'Time_difference_seconds': 'numerical',
            'Customer_Birthyear': 'numerical',
        }

        for column, sdtype in column_sdtypes.items():
            metadata.update_column(
                column_name=column,
                sdtype=sdtype,            
            )

        synthetic_subset = generate_synthetic_data(subset, metadata, n_cls_per_gen)
        all_synthetic_data, all_synthetic_data_submission = postprocess_and_concat(
            synthetic_subset, train, all_synthetic_data, all_synthetic_data_submission
        )
    
    return all_synthetic_data, all_synthetic_data_submission


if __name__ == "__main__":
    # Load data
    train = pd.read_csv('/workspace/Dataset/FSI/train.csv')
    train = train.drop(columns="ID")

    fraud_types = train['Fraud_Type'].unique()
    n_cls_per_gen = 10000
    n_sample = 100

    all_synthetic_data, all_synthetic_data_submission = main_process(train, fraud_types, n_cls_per_gen, n_sample)

    # Save the results
    all_synthetic_data.to_csv('/workspace/Dataset/FSI/generated_data.csv', index=False)
    all_synthetic_data_submission.to_csv('/workspace/Dataset/FSI/generated_data_submission.csv', index=False)