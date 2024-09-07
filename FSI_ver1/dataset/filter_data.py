import pandas as pd
import torch
import torch.nn.functional as F

def sample_group(group):
    return group.sample(n=1000, random_state=42, replace=False)  # Use replace=True to handle cases with less than 10,000 samples

# Load the datasets
all_features_df = pd.read_csv('/workspace/Personal_Development/Dacon_FSI/pretrain/feature_extraction/all_features.csv')
mean_features_df = pd.read_csv('/workspace/Personal_Development/Dacon_FSI/pretrain/feature_extraction/mean_features_by_label.csv')

original_features_df = pd.read_csv('/workspace/Dataset/FSI/generated_data.csv')
submission_df = pd.read_csv('/workspace/Dataset/FSI/generated_data_submission.csv')

# Extract labels and features from all_features_with_labels.csv
all_labels = all_features_df['Fraud_Type']
all_features = all_features_df.drop(columns=['Fraud_Type', 'ID']).values  # Assuming 'ID' and 'Fraud_Type' columns exist

# Extract fraud types and their corresponding mean features
fraud_types = mean_features_df['Fraud_Type']
mean_features = mean_features_df.drop(columns=['Fraud_Type']).values  # Assuming 'Fraud_Type' column exists

# Convert to PyTorch tensors
all_features_tensor = torch.tensor(all_features, dtype=torch.float32)
mean_features_tensor = torch.tensor(mean_features, dtype=torch.float32)

# Dictionary to store sorted indices by cosine similarity for each fraud type
closest_samples_by_fraud_type = {}

# Iterate over each fraud type
for i, fraud_type in enumerate(fraud_types):
    # Filter the samples with the same fraud type
    same_fraud_type_indices = all_labels[all_labels == fraud_type].index
    same_fraud_type_features = all_features_tensor[same_fraud_type_indices]

    # Get the mean feature vector for the current fraud type
    mean_vector = mean_features_tensor[i].unsqueeze(0)  # Shape: (1, feature_dim)
    
    # Compute cosine similarity between filtered samples and the mean vector
    cosine_similarities = F.cosine_similarity(same_fraud_type_features, mean_vector, dim=-1)
    
    # Sort the indices by cosine similarity (highest first)
    sorted_indices = torch.argsort(cosine_similarities, descending=True)
    
    # Store the sorted indices for this fraud type (adjusted to original dataset)
    closest_samples_by_fraud_type[fraud_type] = same_fraud_type_indices[sorted_indices].values

# Save the top 3000 samples for each fraud type in a single file
all_top_samples_df = pd.concat([all_features_df.iloc[closest_samples_by_fraud_type[fraud_type][:5000]] for fraud_type in fraud_types], ignore_index=True)
output_file = '/workspace/Personal_Development/Dacon_FSI/pretrain/feature_extraction/filtered_feature_3000.csv'
all_top_samples_df.to_csv(output_file, index=False)
print(f"Saved top 3000 samples for all fraud types to {output_file}")

# Filter and save submission_df using the top 1000 samples for each fraud type
filtered_submission_dfs = []
for fraud_type in fraud_types:
    top_samples = closest_samples_by_fraud_type[fraud_type]
    # Get the top 1000 indices
    top_indices = top_samples[:1300]
    
    # Print the length to debug
    print(f"Fraud Type: {fraud_type}, Indices Count: {len(top_indices)}")
    
    # Filter submission_df based on these indices using iloc
    filtered_submission_df = submission_df.iloc[top_indices]
    
    # Print the length of the filtered DataFrame to debug
    print(f"Filtered Submission Count for {fraud_type}: {len(filtered_submission_df)}")
    # print(f"{len(top_indices)}")
    # Filter submission_df based on these indices
    
    # Store filtered submission_df for concatenation
    filtered_submission_dfs.append(filtered_submission_df)

# Concatenate all filtered submission_df parts and save to CSV
final_filtered_submission_df = pd.concat(filtered_submission_dfs, ignore_index=True)


final_filtered_submission_df  = final_filtered_submission_df.groupby(final_filtered_submission_df.columns[-3], group_keys=False).apply(sample_group).reset_index(drop=True)
    
submission_output_file = '/workspace/Personal_Development/Dacon_FSI/pretrain/feature_extraction/syn_submission.csv'
final_filtered_submission_df.to_csv(submission_output_file, index=False)
print(f"Saved filtered submission_df to {submission_output_file}")

# Filter and save original_features_df using the top 5000 samples for each fraud type
filtered_original_dfs = []
for fraud_type in fraud_types:
    top_samples = closest_samples_by_fraud_type[fraud_type]
    # Get the top 1000 indices
    top_indices = top_samples[:5000]
    
    # Filter original_features_df based on these indices
    filtered_original_df = original_features_df.iloc[top_indices]
    
    # Store filtered original_df for concatenation
    filtered_original_dfs.append(filtered_original_df)

# Concatenate all filtered original_df parts and save to CSV
final_filtered_original_df = pd.concat(filtered_original_dfs, ignore_index=True)
original_output_file = '/workspace/Personal_Development/Dacon_FSI/pretrain/feature_extraction/filtered_data_3000.csv'
final_filtered_original_df.to_csv(original_output_file, index=False)
print(f"Saved filtered original_features_df to {original_output_file}")