import torch
print("PyTorch - CUDA available:", torch.cuda.is_available())

import xgboost as xgb

# Create a simple DMatrix
dtrain = xgb.DMatrix(data=[[1, 2], [3, 4], [5, 6]], label=[0, 1, 0])

# Train a small model with GPU support
params = {
 'tree_method': 'hist',  # Use the updated method
        'device': 'cuda',       # Specify GPU usage
}

# Train the model
bst = xgb.train(params, dtrain, num_boost_round=10)

# If no errors occurred, XGBoost is using the GPU
print("XGBoost is using the GPU.")



# CatBoost check
from catboost import CatBoostClassifier
# Initialize CatBoost model with GPU usage
model = CatBoostClassifier(task_type='GPU', iterations=10)

# Train on dummy data
model.fit([[0, 1], [1, 0], [0, 0], [1, 1]], [0, 1, 0, 1])

print("CatBoost successfully used the GPU.")

# FastAI check
from fastai.tabular.all import *

# Create a simple DataFrame
df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 0, 1, 0, 1], 'y': [0, 1, 0, 1, 0]})

# Split the data
dls = TabularDataLoaders.from_df(df, y_names='y', cat_names=['a'], cont_names=['b'], procs=[Categorify, Normalize])

# Initialize a simple model
learn = tabular_learner(dls, metrics=accuracy)

# Check if FastAI is using GPU
if torch.cuda.is_available():
    learn.to_fp16()
    print("FastAI is using the GPU.")
else:
    print("FastAI is not using the GPU.")

# LightGBM check
import lightgbm as lgb
import numpy as np

# Create a simple dataset
data = np.array([[1, 2], [3, 4], [5, 6]])  # Convert to NumPy array
label = np.array([0, 1, 0])  # Convert to NumPy array
train_data = lgb.Dataset(data, label=label)

# Define parameters for LightGBM with GPU usage
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',

}

# Train the LightGBM model
gbm = lgb.train(params, train_data, num_boost_round=10)

# If no errors occurred, LightGBM is using the GPU
print("LightGBM is using the GPU.")



import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from autogluon.tabular import TabularPredictor, TabularDataset

## Evaluate the Model & save the preprocessor
import joblib


# Define the root directory
root_dir = '/workspace/Dataset/FSI/'

# Load the datasets
train_df = pd.read_csv(root_dir + 'train_sample.csv')
train_generated_df = pd.read_csv(root_dir + 'submission/filtered_data_3000.csv')
test_df = pd.read_csv(root_dir + 'test.csv')

# Concatenate the training datasets
combined_train_df = pd.concat([train_df, train_generated_df], ignore_index=True)

# Drop the 'ID' column from training data (but not 'Fraud_Type')
combined_train_df = combined_train_df.drop(columns=['ID'])
test_df = test_df.drop(columns=['ID'])


# Check if val.csv exists; if not, split the combined_train_df
val_file_path = root_dir + 'train.csv'
if os.path.exists(val_file_path):
    val_df = pd.read_csv(val_file_path)
    val_df = val_df.drop(columns=['ID'])  # Only drop 'ID', keep 'Fraud_Type' as the label
else:
    combined_train_df, val_df = train_test_split(combined_train_df, test_size=0.2, random_state=42)




# Specify the target column for classification
label = 'Fraud_Type'

# Define hyperparameters for GPU usage
hyperparameters = {
    'NN_TORCH': {
        'use_gpu': True,  # Ensure GPU usage for PyTorch-based models
    },
    'FASTAI': {
        'use_gpu': True,  # Ensure GPU usage for FastAI-based models
    },
    'GBM': [
        {'use_gpu': True, 'device' :'cuda'},  # LightGBM with GPU
        {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}, 'use_gpu': True},  # Extra Trees with GPU
        'GBMLarge',  # AutoGluon will attempt to use GPU if possible
    ],
    'CAT': {
        'task_type': 'GPU',  # Enable GPU for CatBoost
    },
    'XGB': {
    'tree_method': 'hist',  # Use the updated method
            'device': 'cuda',       # Specify GPU usage
    }
}

# Initialize the TabularPredictor
predictor = TabularPredictor(label=label, eval_metric='accuracy' ).fit(
    train_data=TabularDataset(combined_train_df), 
    # tuning_data=TabularDataset(val_df), 
    # presets='good_quality',  # Options: 'best_quality', 'high_quality', 'good_quality', 'medium_quality'
    time_limit = 30,
    hyperparameters=hyperparameters
)


leaderboard = predictor.leaderboard(silent=False)



# Evaluate performance on the training and validation datasets
train_performance = predictor.evaluate(combined_train_df)
val_performance = predictor.evaluate(val_df)

# Display performance metrics
print("Training performance metrics:")
print(train_performance)

print("\nValidation performance metrics:")
print(val_performance)



preprocessor = predictor._trainer._feature_generator
joblib.dump(preprocessor, root_dir + 'preprocessor.pkl')


# Make predictions on the validation set and calculate per-class scores
val_predictions = predictor.predict(val_df)
val_true = val_df[label]
classification_report_per_class = classification_report(val_true, val_predictions, output_dict=True)

# Display classification report for each fraud type
for fraud_type, metrics in classification_report_per_class.items():
    if isinstance(metrics, dict):  # Filter out 'accuracy' and other aggregate scores
        print(f"Fraud Type: {fraud_type}")
        for metric, score in metrics.items():
            print(f"  {metric}: {score}")
        print()
        
        
        
        
# Make predictions on the test set
predictions = predictor.predict(test_df.drop(columns=[label]))

# Prepare the submission file using the ID column from the test data
submission = pd.DataFrame({
    'ID': test_df['ID'],  # Replace 'ID' with the actual ID column name in your test.csv
    'Fraud_Type': predictions
})

# Save the submission file
submission_file_path = root_dir + 'clf_submission.csv'
submission.to_csv(submission_file_path, index=False)

print(f"Submission file saved to {submission_file_path}")