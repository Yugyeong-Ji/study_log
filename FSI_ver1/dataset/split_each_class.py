import pandas as pd

def split_fraud_types(file_path):
    df = pd.read_csv(file_path)

    # Get unique fraud types
    fraud_types = df['Fraud_Type'].unique()

    for fraud_type in fraud_types:
        # Filter the dataframe for the current fraud type
        df_filtered = df[df['Fraud_Type'] == fraud_type]


        # Save to a CSV file
        df_filtered.to_csv(f'fraud_type_{fraud_type}.csv', index=False)
    
    print("Files have been created for each fraud type.")
    
split_fraud_types("/workspace/Dataset/FSI/train.csv")