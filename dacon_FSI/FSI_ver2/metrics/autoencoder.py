import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(args, inputs, reconstructions):
    # Convert tensors to numpy arrays for sklearn metrics
    inputs = inputs.cpu().numpy()
    reconstructions = reconstructions.detach().cpu().numpy()
    
    # Calculate reconstruction error metrics
    mse = mean_squared_error(inputs, reconstructions)
    mae = mean_absolute_error(inputs, reconstructions)
    
    # Calculate per-sample errors
    sample_mse = np.mean((inputs - reconstructions) ** 2, axis=1)
    sample_mae = np.mean(np.abs(inputs - reconstructions), axis=1)
    
    return {
        "mse": mse,
        "mae": mae,
        "sample_mse": sample_mse,
        "sample_mae": sample_mae
    }