
from loss import *

def load_loss(args):
    loss = globals()[args.loss]
    
    print(f"Loss Function :{loss.__name__}")
    
    return loss(args)
