import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import numpy as np

# Load the dataset
file_path = '/workspace/Dataset/FSI/train.csv'
df = pd.read_csv(file_path)

# Clean up the column names by stripping whitespace
df.columns = df.columns.str.strip()

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    if column != 'Fraud_Type' and column != 'ID':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le


# Generate new features
df['Transaction_Velocity'] = df['Number_of_transaction_with_the_account'] / (pd.to_datetime(df['Transaction_resumed_date']) - pd.to_datetime(df['Transaction_Datetime'])).dt.total_seconds()
df['Balance_Change'] = df['Account_balance'] - df['Account_initial_balance']
df['Balance_Transaction_Ratio'] = df['Account_balance'] / df['Transaction_Amount']
df['Transaction_Amount_to_Limit_Ratio'] = df['Transaction_Amount'] / df['Account_amount_daily_limit']
df['Time_Since_Account_Creation'] = (pd.to_datetime(df['Transaction_Datetime']) - pd.to_datetime(df['Account_creation_datetime'])).dt.total_seconds()
df['Geographic_Consistency'] = (df['Location'] == df['IP_Address']).astype(int)


# Define the columns to drop based on previous analysis
columns_to_drop = [
    'Customer_Gender', 'Customer_flag_change_of_authentication_1',
    'Customer_flag_change_of_authentication_2', 'Customer_flag_change_of_authentication_3',
    'Customer_flag_change_of_authentication_4', 'Customer_mobile_roaming_indicator',
    'Customer_VPN_Indicator', 'Customer_flag_terminal_malicious_behavior_1',
    'Customer_flag_terminal_malicious_behavior_2', 'Customer_flag_terminal_malicious_behavior_3',
    'Customer_flag_terminal_malicious_behavior_4', 'Customer_flag_terminal_malicious_behavior_5',
    'Customer_inquery_atm_limit', 'Customer_increase_atm_limit',
    'Account_indicator_release_limit_excess', 'Account_indicator_Openbanking', 'Error_Code',
    'Transaction_Failure_Status', 'Type_General_Automatic', 'Another_Person_Account',
    'Unused_terminal_status', 'Unused_account_status', 'First_time_iOS_by_vulnerable_user'
]

# Drop the unimportant columns
df = df.drop(columns=columns_to_drop)


# Drop any generated columns that result in infinity or NaN values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Save the processed data to a new CSV file
output_file_path = '/workspace/Dataset/FSI/train_preprocess.csv'
df.to_csv(output_file_path, index=False)

print(f"Processed data saved to {output_file_path}")