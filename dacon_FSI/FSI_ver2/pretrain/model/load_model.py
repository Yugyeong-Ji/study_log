import torch
from model import *

def load_model(args):
    model_class = globals()[args.model]
    
    
    if args.test:
        print(f"Testing model: {model_class.__name__}")
    else : 
        print(f"Training model: {model_class.__name__}")
        
    model = model_class(args).to(args.device)
    
    if args.test:
        
        print(f"Loading pretrained weights from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    
    return model
