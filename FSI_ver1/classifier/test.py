
"""Sample train code for the classifier, use it as a template"""

import sys
import os

# change the path to your own directory, where you've cloned / pulled
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn as nn
import torch.optim as optim


from dataset.load_dataset import Load_Dataloader
from tools.yaml_configuration import load_config
from classifier.model.load_model import load_model
from classifier.engine import test_engine



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
    test_dataloader = Load_Dataloader(args)
    
    # Model
    model = load_model(args)

    # Train the model
    test_engine(args, model, test_dataloader)
    
    print(f"Test Complete")
    
    


if __name__ == "__main__":
    
    config_path = os.path.join(os.path.dirname(__file__), 'config',)
    
    config = os.path.join(config_path,'test.yaml')
    
    main(config, device = 1)
    
    # python -m torch.distributed.run --nproc_per_node 4 train.py for ddp