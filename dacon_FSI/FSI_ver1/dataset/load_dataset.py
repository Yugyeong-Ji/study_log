
import os
from dataset import *
from torch.utils.data import  DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split
import numpy as np
import copy

def Load_Dataloader(args):
    
    data_class = load_data(args)
    if args.test:
            
        test_path = os.path.join(args.data_path, args.test_path)
        
        test_dataset = data_class(args, test_path)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        return  test_dataloader
    
    else:
        train_path = os.path.join(args.data_path, args.train_path)
        val_path = os.path.join(args.data_path, args.val_path)

        # If there is a validation dataset, split it 
        if os.path.exists(val_path):
            train_args = copy.deepcopy(args)
            val_args = copy.deepcopy(args)

            # Set the val flag accordingly
            train_args.val = False
            val_args.val = True
            
            train_dataset = data_class(train_args, train_path)
            val_dataset = data_class(val_args, val_path)

        # If there ins't a validation set, split it
        else:
            train_args = copy.deepcopy(args)
            val_args = copy.deepcopy(args)
            
            # Load the full dataset with train_args
            train_args.val = False
            full_dataset = data_class(train_args, train_path)
            train_indices, val_indices = random_split(full_dataset, args.split_ratio)

            # Create train and validation datasets with separate args
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(copy.deepcopy(full_dataset), val_indices)
            
            # Set the val flag for validation dataset
            val_dataset.dataset.args.val = True

        # DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        
        return train_dataloader, val_dataloader



def random_split(dataset, split_ratio):
    # Extract indices of the dataset
    indices = np.arange(len(dataset))
    
    # Perform random split
    train_indices, val_indices = train_test_split(
        indices,
        test_size=1 - split_ratio,
        random_state=42
    )
    
    return train_indices, val_indices


def load_data(args):
    data = globals()[args.dataset]
    
    print(f"Dataset Class : {data.__name__}")
    
    return data
