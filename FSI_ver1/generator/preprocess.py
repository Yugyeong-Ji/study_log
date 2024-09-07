import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import warnings
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


def handle_outliers(series, n_std=3):
    mean = series.mean()
    std = series.std()
    z_scores = np.abs(stats.zscore(series, nan_policy='omit'))  # Handle NaNs in z-score calculation
    return series.mask(z_scores > n_std, mean)


def preprocess_data(train):
    
    # Convert and handle outliers
    train['Time_difference_seconds'] = pd.to_timedelta(train['Time_difference']).dt.total_seconds()
    train['Time_difference_seconds'] = handle_outliers(train['Time_difference_seconds'])


    date_columns = [
        'Customer_registration_datetime',
        'Account_creation_datetime',
        'Transaction_Datetime',
        'Last_atm_transaction_datetime',
        'Last_bank_branch_transaction_datetime',
        'Transaction_resumed_date'
    ]
    for column in date_columns:
        train[column] = pd.to_datetime(train[column])
        
    train['Diff_Customer_to_Account'] = (train['Account_creation_datetime'] - train['Customer_registration_datetime']).dt.total_seconds()
    train['Diff_Account_to_Transaction'] = (train['Transaction_Datetime'] - train['Account_creation_datetime']).dt.total_seconds()
    train['Diff_LastATM_to_Transaction'] = (train['Transaction_Datetime'] - train['Last_atm_transaction_datetime']).dt.total_seconds()
    train['Diff_LastBank_to_Transaction'] = (train['Transaction_Datetime'] - train['Last_bank_branch_transaction_datetime']).dt.total_seconds()
    train['Diff_Transaction_to_TransactionResumed'] = (train['Transaction_Datetime'] - train['Transaction_resumed_date']).dt.total_seconds()
    

    return train


def create_metadata(subset):
            # Create and update metadata for each fraud type subset
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(subset)
        metadata.set_primary_key(None)
        
        # Ensure the data types are set correctly
        column_sdtypes = {

            'Fraud_Type': 'categorical',
            'Time_difference_seconds': 'numerical',
            
            'Diff_Customer_to_Account': 'numerical',
            'Diff_Account_to_Transaction': 'numerical',
            'Diff_LastATM_to_Transaction': 'numerical',
            'Diff_LastBank_to_Transaction': 'numerical',
            'Diff_Transaction_to_TransactionResumed': 'numerical',
            

        }

        for column, sdtype in column_sdtypes.items():
            metadata.update_column(
                column_name=column,
                sdtype=sdtype,            
            )
            
            
        return metadata