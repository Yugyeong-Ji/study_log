import pandas as pd

# Load the generated data
generated_data = pd.read_csv('/workspace/Dataset/FSI/generated_data.csv')

# Remove all samples with Fraud_Type "m"
generated_data = generated_data[generated_data['Fraud_Type'] != 'm']

# Load the original dataset
original_data = pd.read_csv('/workspace/Dataset/FSI/train_sample.csv')

# Select 10,000 samples with Fraud_Type "m"
m_samples = original_data[original_data['Fraud_Type'] == 'm'].sample(n=10000, random_state=42)

# Initialize an empty DataFrame for the other sampled fraud types
n_samples = pd.DataFrame()

# Get a list of unique fraud types excluding 'm'
fraud_types = original_data['Fraud_Type'].unique()
fraud_types = fraud_types[fraud_types != 'm']

# Loop through each fraud type and sample n=100
for fraud_type in fraud_types:
    sample = original_data[original_data['Fraud_Type'] == fraud_type].sample(n=100, random_state=42)
    n_samples = pd.concat([n_samples, sample], ignore_index=True)

# Concatenate the remaining generated data, the selected "m" samples, and the n_samples
final_data = pd.concat([generated_data, m_samples, n_samples], ignore_index=True)

# Save the final dataset
final_data.to_csv('/workspace/Dataset/FSI/final_training_data.csv', index=False)

print("Final dataset saved as 'final_training_data.csv'")