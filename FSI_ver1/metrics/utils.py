import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import joblib

def print_score(data_split: str, scores: dict):
    try:
        print(f"{data_split} : ")
        max_key_length = max(len(key) for key in scores.keys())
        
        for key, value in scores.items():
            if isinstance(value, (list, np.ndarray)):
                print(f'{key.capitalize():<{max_key_length}} :')
                for i, v in enumerate(value):
                    print(f'  Class {i:<{max_key_length}} : {v:.4f}')
            else:
                print(f'{key.capitalize():<{max_key_length}} : {value:.4f}')
    except:
        pass
            
            

def save_labels_and_predictions(args, labels, predictions):
    try:
        labels = labels.numpy()
        predictions = predictions.detach().numpy()
        
        # Assuming 'le_subclass' is the LabelEncoder used in preprocess
        # If it's not available here, you should save and reuse it properly
        # le_subclass = LabelEncoder()

        le_subclass = joblib.load(os.path.join(args.save_path,'label_encoder.pkl'))

        original_labels = le_subclass.inverse_transform(labels.astype(int))
        original_predictions = le_subclass.inverse_transform(predictions.argmax(axis=1).astype(int))
        
        df = pd.DataFrame({
            'Original Labels': original_labels,
            'Predictions': original_predictions
        })
        
        df.to_csv(os.path.join(args.save_path, 'labels_predictions.csv'), index=False)
    except:
        pass
    
    

def save_submission_csv(args, ids, predictions):
    
    # Ensure predictions are detached from the computation graph and converted to numpy array
    predictions = predictions.detach().numpy()
    # Assuming 'le_subclass' is the LabelEncoder used in preprocess
    pkl_path = os.path.dirname(args.model_path)
    le_subclass = joblib.load(os.path.join(pkl_path, 'label_encoder.pkl'))

    # Convert predictions to original labels
    original_predictions = le_subclass.inverse_transform(predictions.argmax(axis=1).astype(int))

    # Create a DataFrame with IDs and predicted fraud types
    df = pd.DataFrame({
        'ID': ids,
        'Fraud_Type': original_predictions
    })
    
    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(args.save_path, 'clf_predictions.csv'), index=False)

# Example usage
# Assuming `args` is defined and contains `save_path`
# `ids` is a list of IDs and `predictions` is a tensor of model predictions
