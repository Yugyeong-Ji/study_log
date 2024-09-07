import pandas as pd

# Load the dataset
data = pd.read_csv('/workspace/Dataset/FSI/generated_data.csv')

# Print column names, number of rows, and number of columns
print("Columns:", data.columns.tolist())
print("Total number of Rows:", len(data))
print("Total number of Columns:", len(data.columns))

# Iterate over the columns and print their names
for i in range(len(data.columns)):
    print(f"{i}th column: {data.columns[i]}, dtype: {data.dtypes.iloc[i]}")

if len(data.columns) >= 1:
    # Find the index of the "Fraud_Type" column
    fraud_type_index = data.columns.get_loc("Fraud_Type")
    print(f"Fraud_Type column index: {fraud_type_index}")

    # Get value counts for the "Fraud_Type" column
    column_values_count = data.iloc[:, fraud_type_index].value_counts()
    total_distinct_values = len(column_values_count)
    print(f"Value counts for the Fraud_Type column: {total_distinct_values}") 
    print(column_values_count)
    print("Total number of distinct values in the Fraud_Type column:", total_distinct_values)
    
    # Sample 10,000 instances for each label in the "Fraud_Type" column
    def sample_group(group):
        return group.sample(n=2000, random_state=532, replace=True)  # Use replace=True to handle cases with less than 10,000 samples

    sampled_data = data.groupby(data.columns[fraud_type_index], group_keys=False).apply(sample_group).dropna().reset_index(drop=True)
    
    
    # Save the sampled data to a new CSV file
    sampled_data.to_csv("/workspace/Dataset/FSI/generated_sample.csv", index=False)
    print("Sampled data saved to /workspace/Dataset/FSI/train_sample_preprocessed.csv")