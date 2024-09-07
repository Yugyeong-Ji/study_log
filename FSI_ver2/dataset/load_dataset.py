
import os
from dataset import *
from torch.utils.data import  DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split
import numpy as np

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
            args.val= False
            train_dataset =  data_class(args, train_path)
            args.val = True
            val_dataset =  data_class(args, val_path)


        # If there ins't a validation set, split it
        else:
            args.val = False
            full_dataset =  data_class(args, train_path)
            train_indices, val_indices = random_split(full_dataset, args.split_ratio)

            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)

        # DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        # DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        
        
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
