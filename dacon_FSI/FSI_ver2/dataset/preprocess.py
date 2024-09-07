import pandas as pd

def load_and_preprocess_data(data, output_file_path):

    # Convert relevant columns to numeric, forcing errors to NaN
    data['Distance'] = pd.to_numeric(data['Distance'], errors='coerce')
    data['Time_difference'] = pd.to_numeric(data['Time_difference'], errors='coerce')
    
    # Derive new columns
    # 1. Transaction Frequency: Count of transactions per day
    data['Transaction_Date'] = pd.to_datetime(data['Transaction_Datetime']).dt.date
    data['Transaction_Frequency'] = data.groupby('Transaction_Date')['Transaction_Datetime'].transform('count')
    
    # 2. Average Transaction Amount per day
    data['Avg_Transaction_Amount'] = data.groupby('Transaction_Date')['Transaction_Amount'].transform('mean')
    
    # 3. Time Since Last Transaction
    data['Transaction_Datetime'] = pd.to_datetime(data['Transaction_Datetime'])
    data = data.sort_values(by=['Transaction_Datetime'])
    data['Time_Since_Last_Transaction'] = data.groupby('Account_account_number')['Transaction_Datetime'].diff().dt.total_seconds()
    
    # 4. Account Activity Score (simple version as sum of transactions)
    data['Account_Activity_Score'] = data.groupby('Account_account_number')['Transaction_Amount'].transform('sum')
    
    # 5. Distance vs. Time Ratio (adding a small value to avoid division by zero)
    data['Distance_vs_Time_Ratio'] = data['Distance'] / (data['Time_difference'] + 1e-5)
    
    # Fill any NaN values that might have been generated
    data = data.fillna(0)
    
    # Drop non-relevant columns based on importance analysis
    columns_to_drop = [# 'ID',  # Need for test
        'Customer_personal_identifier', 'Customer_identification_number', 
        'Account_account_number', 'IP_Address', 'MAC_Address', 'Recipient_Account_Number',
        'Another_Person_Account', 'First_time_iOS_by_vulnerable_user', 
        'Error_Code', 'Transaction_Failure_Status'
    ]
    
    # Drop the identified columns
    data = data.drop(columns=columns_to_drop, errors='ignore')
    
    # Drop intermediate columns that are no longer necessary
    data = data.drop(columns=['Transaction_Date'], errors='ignore')
    
    for i, col in enumerate(data.columns, 1):
        print(f"Column {i}: {col}")
    
    if output_file_path:
        # Save the processed DataFrame to a new CSV file
        data.to_csv(output_file_path, index=False)
        
    return data

data = pd.read_csv("/workspace/Dataset/FSI/generated_data.csv" )
load_and_preprocess_data(data, "/workspace/Dataset/FSI/generated_data_preprocessed.csv")