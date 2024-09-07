import yaml
import os
import shutil
import torch
import random
import numpy as np


import sys
import logging

def setup_logging(save_path, log_filename='output.log'):
    log_file = os.path.join(save_path, log_filename)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)  # Optional: Comment out this line if you don't want to print to console
        ]
    )
    logging.info(f"Logging to {log_file}")

class Yaml_Configuration:
    def __init__(self, config_dict):

        for key, value in config_dict.items():
            setattr(self, key, value)


def load_config(configure_path: str, device : int = 1):
    # Load the YAML configuration file
    with open(configure_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    # Convert dictionary to Config object
    config = Yaml_Configuration(config_dict)
    
    try:
        if not os.path.exists(config.feature_path):
            os.makedirs(config.feature_path)
    except: 
        pass
    if config.test :
        config.save_path = create_save_dir(config, yaml_file=configure_path, base_dir ="submission")
    else:
        config.save_path = create_save_dir(config, yaml_file=configure_path)
    config.device = configure_device(device)
    set_all_seeds(config.random_seed)
    
    
    setup_logging(config.save_path)
    
    
    return config



def create_save_dir( args, yaml_file : str, base_dir='run'):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    run_id = 1
    if args.test:
        while os.path.exists(os.path.join(base_dir, f'test{run_id}')):
            run_id += 1
        
        save_dir = os.path.join(base_dir, f'test{run_id}')
        os.makedirs(save_dir)
        
        print(f"the results will be saved in {save_dir}")
        
    else:
        while os.path.exists(os.path.join(base_dir, f'train{run_id}')):
            run_id += 1
        
        save_dir = os.path.join(base_dir, f'train{run_id}')
        os.makedirs(save_dir)
        
        print(f"the model will be saved in {save_dir}")
    
    shutil.copy(yaml_file, os.path.join(save_dir, "args.yaml"))
    
    return save_dir


def configure_device(device):
    if device == 1:
        if torch.cuda.is_available():
            print("Using CUDA")
            return 'cuda'
        elif torch.backends.mps.is_available():
            print("Using MPS, MacBook Perhaps?")
            return 'mps'
        else:
            print("Using CPU, Check if device is correctly used")
            return 'cpu'
    elif device > 1:
        if torch.cuda.device_count() >= device:
            print("Using Multiple GPUs")
            return [f'cuda:{i}' for i in range(device)]
        else:
            print("Not enough GPUs available.")
            return 'cpu'


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
