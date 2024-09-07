
from .classweight_CE import  WeightedCrossEntropyLoss as classweight_CE
from .original_CE import CrossEntropyLoss as  original_CE
from .dynamic_weight_CE import DynamicWeightedCrossEntropyLoss as dynamic_weight_CE
from .MSE import MSELoss as MSE
from .contrastive_loss import ContrastiveLoss as contrastive_loss
from .focal_loss import FocalLoss as focal_loss
from .e_contrastive_loss import EuclideanContrastiveLoss as e_contrastive_loss
__all__ = ["classweight_CE",
           "original_CE",
           "dynamic_weight_CE",
           "MSE",
           "contrastive_loss",
           "focal_loss",
           "e_contrastive_loss",
           ]

