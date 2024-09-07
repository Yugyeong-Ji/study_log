# Define the neural network
from .modules import *
import torch
import torch.nn as nn
import numpy as np

class FSI_Transformer_Autoencoder(nn.Module):
    def __init__(self, args):
        super(FSI_Transformer_Autoencoder, self).__init__()
        
        self.args = args

        self.feature_weights = nn.Parameter(torch.ones(args.input_size))
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(128, args.input_size)
        
        # Encoder
        self.input_projection = MLP(input_dim=args.input_size, hidden_dim=256, output_dim=128 * args.input_size)
        self.input_bn = nn.BatchNorm1d(128)
        self.transformer_encoder = nn.Sequential(
            *[TransformerLayer(input_dimension=128) for _ in range(args.blocks)]
        )
        
        # Decoder
        self.output_projection = MLP(input_dim=128 * args.input_size, hidden_dim=256, output_dim=args.input_size)

    def forward(self, x):
        # Input x shape: (batch_size, input_size)
        
        # Apply feature weights
        x = x * self.feature_weights
        
        # Encoder
        x = self.input_projection(x)  # Shape: (batch_size, input_size * 128)
        x = x.view(x.size(0), -1, 128)  # Reshape for BatchNorm1d
        x = self.input_bn(x.transpose(1, 2)).transpose(1, 2)  # Shape: (batch_size, input_size, 128)
        
        # Apply masking after projection
        mask = self._create_mask(x.size(), self.args.mask_ratio).to(x.device)
        x = x * mask
        
        # Positional encoding
        x = self.positional_encoding(x)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)  # Shape: (batch_size, input_size, 128)
        
        # Decoder
        x = x.reshape(x.size(0), -1)
        x = self.output_projection(x)  # Shape: (batch_size, input_size)
        
        return x
    
    def _create_mask(self, shape, mask_ratio):
        """
        Create a binary mask for the input tensor.

        Args:
            shape (torch.Size): The shape of the input tensor.
            mask_ratio (float): The ratio of elements to mask.

        Returns:
            torch.Tensor: The binary mask.
        """
        batch_size, seq_len, feature_dim = shape
        num_masked = int(mask_ratio * seq_len)
        mask = torch.ones(shape)
        
        for i in range(batch_size):
            masked_indices = np.random.choice(seq_len, num_masked, replace=False)
            mask[i, masked_indices, :] = 0
        
        return mask