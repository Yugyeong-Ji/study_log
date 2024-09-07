import pandas as pd
from tqdm import tqdm
from sdv.single_table import CTGANSynthesizer
import warnings

from preprocess import *
from augmentation import *
from postprocess import *

# Ignore specific warnings
warnings.filterwarnings('ignore', message="There is an existing primary key 'ID'. This key will be removed.")
warnings.filterwarnings('ignore', message="We strongly recommend saving the metadata using 'save_to_json' for replicability in future SDV versions.")


def generate_synthetic_data(subset, metadata, n_cls_per_gen, total_epochs=800, batch_size=500):
    
    synthesizer = CTGANSynthesizer(metadata, epochs=total_epochs, batch_size=batch_size, cuda=True)
    synthesizer.fit(subset)

    # Sample data after training
    synthetic_subset = synthesizer.sample(n_cls_per_gen)
    

    
    synthetic_subset['Time_difference_seconds'] = handle_outliers(synthetic_subset['Time_difference_seconds'])
    synthetic_subset['Time_difference'] = pd.to_timedelta(synthetic_subset['Time_difference_seconds'], unit='s')
    synthetic_subset['Account_creation_datetime'] = synthetic_subset['Customer_registration_datetime'] + pd.to_timedelta(synthetic_subset['Diff_Customer_to_Account'], unit='s')
    synthetic_subset['Transaction_Datetime'] = synthetic_subset['Account_creation_datetime'] + pd.to_timedelta(synthetic_subset['Diff_Account_to_Transaction'], unit='s')
    synthetic_subset['Last_atm_transaction_datetime'] = synthetic_subset['Transaction_Datetime'] - pd.to_timedelta(synthetic_subset['Diff_LastATM_to_Transaction'], unit='s')
    synthetic_subset['Last_bank_branch_transaction_datetime'] = synthetic_subset['Transaction_Datetime'] - pd.to_timedelta(synthetic_subset['Diff_LastBank_to_Transaction'], unit='s')
    synthetic_subset['Transaction_resumed_date'] = synthetic_subset['Transaction_Datetime'] - pd.to_timedelta(synthetic_subset['Diff_Transaction_to_TransactionResumed'], unit='s')
    
    synthetic_subset = synthetic_subset.drop('Time_difference_seconds', axis=1)
    synthetic_subset = synthetic_subset.drop(columns=[
    'Diff_Customer_to_Account',
    'Diff_Account_to_Transaction',
    'Diff_LastATM_to_Transaction',
    'Diff_LastBank_to_Transaction',
    'Diff_Transaction_to_TransactionResumed'
    ], axis=1)

    return synthetic_subset


def main_process(train, fraud_types, n_cls_per_gen, n_sample, epochs=300, batch_size=500):
    synthetic_each_csv = pd.DataFrame()

    for fraud_type in tqdm(fraud_types):
        
        subset = train[train["Fraud_Type"] == fraud_type]
        # Preprocess & Sampling
        subset = subset.sample(n=n_sample, replace= True)


        # Drop the original 'Time_difference' column (use only the seconds version)
        subset = subset.drop('Time_difference', axis=1)
        subset = subset.drop(columns=[
        'Account_creation_datetime',
        'Transaction_Datetime',
        'Last_atm_transaction_datetime',
        'Last_bank_branch_transaction_datetime',
        'Transaction_resumed_date'
        ], axis=1)
        # Create Metadata
        metadata = create_metadata(subset)

        
        # Generate synthetic data and save the model
        synthetic_subset = generate_synthetic_data(subset, metadata, n_cls_per_gen, epochs, batch_size)
        synthetic_each_csv = pd.concat([synthetic_each_csv,synthetic_subset],axis=0)

    return synthetic_each_csv


def split_csv(df):
    # List of columns considered as irrelevant
    unrelavent_columns = [
        'Customer_personal_identifier',
        'Customer_identification_number',
        'Error_Code', 
        'Type_General_Automatic',
        'IP_Address',
        'MAC_Address',
        'Location',
        'Recipient_Account_Number',
        'Another_Person_Account',
        'First_time_iOS_by_vulnerable_user'
    ]
    
    # Create a DataFrame with only the irrelevant columns
    train_unrelavent = df[unrelavent_columns + ['Fraud_Type']]
    
    # Drop these columns from the original DataFrame
    df_relevant = df.drop(columns=unrelavent_columns, errors='ignore')
    
    # Group the relevant columns based on the categories identified earlier
    
    # Customer Information
    customer_info_cols = [
        'Customer_Birthyear', 'Customer_Gender',
        'Customer_registration_datetime', 'Customer_credit_rating', 
        'Customer_flag_change_of_authentication_1', 'Customer_flag_change_of_authentication_2',
        'Customer_flag_change_of_authentication_3', 'Customer_flag_change_of_authentication_4', 
        'Customer_loan_type', 'Fraud_Type'
    ]
    customer_info = df_relevant[customer_info_cols]
    
    # Account and Transaction Information
    account_info_cols = [
        'Account_creation_datetime', 'Account_initial_balance', 'Account_balance',
        'Transaction_Amount', 'Channel', 'Transaction_Failure_Status', 
        'Account_indicator_Openbanking', 'Account_amount_daily_limit',
        'Account_remaining_amount_daily_limit_exceeded', 'Account_release_suspention', 
        'Account_account_type','Account_account_number', 'Fraud_Type'
    ]
    account_info = df_relevant[account_info_cols]
    
    # Security and Risk Factors
    security_risk_cols = [
        'Customer_rooting_jailbreak_indicator', 'Customer_mobile_roaming_indicator', 
        'Customer_VPN_Indicator', 'Customer_flag_terminal_malicious_behavior_1',
        'Customer_flag_terminal_malicious_behavior_2', 'Customer_flag_terminal_malicious_behavior_3',
        'Customer_flag_terminal_malicious_behavior_4', 'Customer_flag_terminal_malicious_behavior_5',
        'Customer_flag_terminal_malicious_behavior_6', 'Recipient_account_suspend_status',
        'Last_atm_transaction_datetime', 'Last_bank_branch_transaction_datetime', 'Fraud_Type'
    ]
    security_risk = df_relevant[security_risk_cols]
    
    # Authentication and Verification
    auth_verification_cols = [
        'Customer_flag_change_of_authentication_1', 'Customer_flag_change_of_authentication_2',
        'Customer_flag_change_of_authentication_3', 'Operating_System', 
        'Access_Medium', 'Transaction_num_connection_failure', 'Distance', 'Fraud_Type'
    ]
    auth_verification = df_relevant[auth_verification_cols]
    
    # Special Transaction Flags
    special_flags_cols = [
        'Customer_inquery_atm_limit', 'Customer_increase_atm_limit', 'Account_indicator_release_limit_excess',
        'Account_one_month_max_amount', 'Account_one_month_std_dev',
        'Account_dawn_one_month_max_amount', 'Account_dawn_one_month_std_dev',
        'Unused_terminal_status', 'Flag_deposit_more_than_tenMillion', 
        'Unused_account_status', 'Number_of_transaction_with_the_account', 
        'Transaction_history_with_the_account', 'Transaction_resumed_date', 'Fraud_Type'
    ]
    special_flags = df_relevant[special_flags_cols]
    
    # DateTime Columns 
    datetime_cols = [
        'Transaction_Datetime', 'Customer_registration_datetime', 'Account_creation_datetime', 
        'Last_atm_transaction_datetime', 'Last_bank_branch_transaction_datetime', 
        'Transaction_resumed_date', 'Time_difference', 'Fraud_Type'
    ]
    datetime_info = df_relevant[datetime_cols]

    return [customer_info, account_info, security_risk, auth_verification, special_flags, train_unrelavent], datetime_info



if __name__ == "__main__":
    
    all_synthetic_data = pd.DataFrame()
    
    # Load data
    original = pd.read_csv('/workspace/Dataset/FSI/train.csv')
    
    original = original.drop(columns="ID")
    
    fraud_types =  original['Fraud_Type'].unique()
    
    group_list, date = split_csv(original)
    
    
    # All Groups
    for csv in group_list:
        synthetic_each_csv = pd.DataFrame()
        for fraud_type in tqdm(fraud_types):
            subset = csv[csv["Fraud_Type"] == fraud_type]
            # Preprocess & Sampling
            subset = subset.sample(n=100, replace= True)

            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(subset)
            metadata.set_primary_key(None)
            
            column_sdtypes = {
                'Account_initial_balance': 'numerical',
                'Account_balance': 'numerical',
                'Customer_identification_number': 'categorical',  
                'Customer_personal_identifier': 'categorical',
                'Account_account_number': 'categorical',
                'IP_Address': 'ipv4_address',  
                'Location': 'categorical',
                'Recipient_Account_Number': 'categorical',
                'Customer_Birthyear': 'numerical',
            }
            
            for column, sdtype in column_sdtypes.items():
                try:
                    metadata.update_column(
                        column_name=column,
                        sdtype=sdtype,            
                    )
                except: 
                    pass
                
            synthesizer = CTGANSynthesizer(metadata, epochs=300, batch_size=500, cuda=True)
            synthesizer.fit(subset)

            # Sample data after training
            synthetic_subset = synthesizer.sample(10000)
            synthetic_each_csv = pd.concat([synthetic_each_csv, synthetic_subset], axis=0)

        
        
        overlapping_columns = all_synthetic_data.columns.intersection(synthetic_each_csv.columns)

        # Drop overlapping columns from the new data
        synthetic_each_csv = synthetic_each_csv.drop(columns=overlapping_columns)

        # Concatenate the DataFrames column-wise
        all_synthetic_data = pd.concat([all_synthetic_data, synthetic_each_csv], axis=1)

        
    # Date
    train = preprocess_data(date)

    fraud_types = train['Fraud_Type'].unique()
    
    n_cls_per_gen = 10000
    n_sample = 100

    synthetic_subset = main_process(train, fraud_types, n_cls_per_gen, n_sample)

    print(f"Training data done")
    
    overlapping_columns = all_synthetic_data.columns.intersection(synthetic_subset.columns)

    # Drop overlapping columns from the new data
    synthetic_subset = synthetic_subset.drop(columns=overlapping_columns)

    # Concatenate the DataFrames column-wise
    all_synthetic_data = pd.concat([all_synthetic_data, synthetic_subset], axis=1)

    all_synthetic_data  = all_synthetic_data[original.columns]
    
    if 'Time_difference' in all_synthetic_data.columns:
            columns = [col for col in all_synthetic_data.columns if col != 'Time_difference']
            all_synthetic_data_submission = all_synthetic_data[columns + ['Time_difference']]
            
    # Save the results
    all_synthetic_data.to_csv('/workspace/Dataset/FSI/generated_data.csv', index=False)
    all_synthetic_data_submission.to_csv('/workspace/Dataset/FSI/generated_data_submission.csv', index=False)