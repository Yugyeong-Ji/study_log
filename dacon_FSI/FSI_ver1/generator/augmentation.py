
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


def random_augmentation(df, nan_probability=0.1):
    """
    Introduce random NaN values or shuffle categorical features in the DataFrame.
    
    Parameters:
    - df: pd.DataFrame, the DataFrame to augment.
    - nan_probability: float, the probability of setting a value to NaN.
    
    Returns:
    - df_augmented: pd.DataFrame, the augmented DataFrame.
    """
    df_augmented = df.copy()

    # Randomly assign NaN values based on the given probability
    mask_nan = np.random.rand(*df_augmented.shape) < nan_probability
    df_augmented[mask_nan] = np.nan

    # Shuffle categorical values for augmentation
    #for column in df_augmented.select_dtypes(include=['object']).columns:
    
    # Shuffl all
    for column in df_augmented.columns:
        mask_shuffle = np.random.rand(df_augmented.shape[0]) < 0.3
        df_augmented.loc[mask_shuffle, column] = np.random.permutation(df_augmented.loc[mask_shuffle, column])

    return df_augmented

def oversample_with_augmentation(train, target_column='Fraud_Type', n_cls_per_gen=1000, nan_probability=0.1):
    """
    Perform oversampling with random augmentation for each class and concatenate with original data.
    
    Parameters:
    - train: pd.DataFrame, the input training dataset.
    - target_column: str, the name of the target column to balance.
    - n_cls_per_gen: int, the number of samples to generate per class.
    - nan_probability: float, the probability of introducing NaN values in augmented data.
    
    Returns:
    - df_combined: pd.DataFrame, the combined original and augmented dataset.
    """
    synthetic_data_list = []

    for fraud_type in train[target_column].unique():
        df_class = train[train[target_column] == fraud_type]
        
        if len(df_class) >= n_cls_per_gen:
            # If the class already has enough samples, add the original data without modification
            synthetic_data_list.append(df_class)
            continue

        # Number of samples needed to reach the desired amount
        n_samples_needed = n_cls_per_gen - len(df_class)
        
        # Oversample the class without altering the target column
        samples = df_class.sample(n=n_samples_needed, replace=True)
        samples_augmented = random_augmentation(samples.drop(columns=[target_column]), nan_probability=nan_probability)
        
        # Reattach the target column to the augmented data
        samples_augmented[target_column] = fraud_type

        # Add the original and augmented data for this class
        synthetic_data_list.append(df_class)  # Keep the original data
        synthetic_data_list.append(samples_augmented)  # Add the augmented samples

    # Combine the list of DataFrames into one DataFrame
    df_synthetic = pd.concat(synthetic_data_list, ignore_index=True)

    # Reset index and save to CSV
    df_synthetic = df_synthetic.reset_index(drop=True)
    df_synthetic.to_csv('/workspace/Dataset/FSI/oversampling.csv', index=False)
    
    return df_synthetic

# def apply_smote(train, target_column='Fraud_Type'):
#     """
#     Apply SMOTE to the given dataset to handle class imbalance.
    
#     Parameters:
#     - train: pd.DataFrame, the input training dataset.
#     - target_column: str, the name of the target column to balance using SMOTE.
    
#     Returns:
#     - df_resampled: pd.DataFrame, the resampled dataset with balanced classes.
#     """
    
#     # Step 1: Encode categorical variables
#     label_encoders = {}
#     for column in train.select_dtypes(include=['object']).columns:
#         if column != target_column:  # Skip target variable
#             le = LabelEncoder()
#             train[column] = le.fit_transform(train[column])
#             label_encoders[column] = le

#     # Step 2: Separate features and target variable
#     X = train.drop(columns=[target_column, 'ID'])  # Drop target and ID column
#     y = train[target_column]

#     # Encode target variable
#     y = LabelEncoder().fit_transform(y)

#     # Step 3: Apply SMOTE
#     smote = SMOTE(random_state=42)
#     X_res, y_res = smote.fit_resample(X, y)

#     # Step 4: Combine resampled data into a new DataFrame
#     df_resampled = pd.DataFrame(X_res, columns=X.columns)
#     df_resampled[target_column] = y_res
    
#     return df_resampled
    
