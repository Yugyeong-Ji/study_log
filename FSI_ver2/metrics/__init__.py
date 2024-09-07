from .autoencoder import evaluate as evaluate_autoencoder
from .binary_classification import evaluate as evaluate_binary_classification
from .multiclass_classification import evaluate as evaluate_multiclass_classification
from .clip_evaluation import evaluate as evaluate_clip
# Dictionary to map metric names to their respective evaluate functions
evaluate_functions = {
    'autoencoder': evaluate_autoencoder,
    'binary_classification': evaluate_binary_classification,
    'multiclass_classification': evaluate_multiclass_classification,
    'clip_evaluation': evaluate_clip
}

def load_evaluate(args):

    if args.metric in evaluate_functions:
        return evaluate_functions[args.metric]
    else:
        raise ValueError("Invalid argument: specify one of 'autoencoder', 'binary_classification', 'clip_evaluation' or 'multiclass_classification'")
    

def load_metric(args):
    metric = load_evaluate(args)
    
    return metric