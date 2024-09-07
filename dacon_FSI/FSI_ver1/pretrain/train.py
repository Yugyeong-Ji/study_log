
"""Sample train code for the classifier, use it as a template"""

import sys
import os

# change the path to your own directory, where you've cloned / pulled
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append("/workspace/Dacon_FSI/") # DBKim Server
import torch
import torch.optim as optim


from dataset.load_dataset import Load_Dataloader
from tools.yaml_configuration import load_config
from model.load_model import load_model
from engine import train_engine
from loss.load_loss import load_loss

# from loss.loss import Loss_Function
# from optimizer import 

def main(configure_path : str, device : int =  1):
    
    """
    configure_path : yaml file that consists configrations
    
    device : number of gpus ( or cpu) to use. 1 in default, if > 1, DDP ( To Do)
    """

    # configuration
    args = load_config(configure_path, device)
    
    # Dataset
    train_dataloader, val_dataloader = Load_Dataloader(args)
    
    # Model
    model = load_model(args)

    # Loss Function
    criterion = load_loss(args)
    
    # Optimizer
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Train the model
    train_engine(args, model, train_dataloader, val_dataloader, criterion, optimizer, scheduler)
    
    print(f"Training Complete")
    


if __name__ == "__main__":
    print(f"a")
    config_path = os.path.join(os.path.dirname(__file__), 'config',)
    
    config = os.path.join(config_path,'clip_train.yaml')
    # config = os.path.join(config_path,'autoencoder_train.yaml')
    
    main(config, device = 1)
    
    # python -m torch.distributed.run --nproc_per_node 4 train.py for ddp