from .simple_model import SimpleNN as simple_model
from .model_v1 import FSI_Transformer as model_v1
from .model_v2 import FSI_Transformer2 as model_v2

__all__ = ["simple_model",
           "model_v1",
           "model_v2",
           ]

