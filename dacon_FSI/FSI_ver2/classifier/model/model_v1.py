# Define the neural network
from classifier.model.modules import *
import torch
import torch.nn as nn

class FSI_Transformer(nn.Module):
    def __init__(self, args):
        super(FSI_Transformer, self).__init__()
        
        self.feature_weights = nn.Parameter(torch.ones(args.input_size))
        
        # Project the input to the desired dimension using MLP
        self.input_projection = MLP(input_dim=args.input_size, hidden_dim=256, output_dim=128 * args.input_size)
        
        # postional encoding
        self.positional_encoding = PositionalEncoding(128, args.input_size)
        
        # Batch normalization for the projected input
        self.input_bn = nn.BatchNorm1d(128)
        
        # Transformer encoder with multiple layers
        self.transformer_encoder = nn.Sequential(
            *[TransformerLayer(input_dimension=128) for _ in range(args.blocks)]
        )
        
        # Classification head using MLP
        self.classification_head = MLP(input_dim=128 * args.input_size, hidden_dim=256, output_dim=args.num_classes)
        
        # Batch normalization for the classification head input
        self.head_bn = nn.BatchNorm1d(128 * args.input_size)
        
    def forward(self, x):
        # Input x shape: (batch_size, input_size)
        
        # x = x * self.feature_weights
        
        # Project input to (batch_size, input_size, 128)
        x = self.input_projection(x)  # Shape: (batch_size, input_size, 128)
        
        
        # Apply batch normalization to the projected input
        x = x.view(x.size(0), -1, 128)  # Reshape for BatchNorm1d
        x = self.input_bn(x.transpose(1, 2)).transpose(1, 2)  # Shape: (batch_size, input_size, 128)
        
        # Positional Encoding
        x = self.positional_encoding(x)
         
         
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # Shape: (batch_size, input_size, 128)
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)  # Shape: (batch_size, input_size * 128)
        
        # Apply batch normalization to the flattened output
        x = self.head_bn(x)
        
        # Classification head
        x = self.classification_head(x)  # Shape: (batch_size, num_classes)
        
        return x
    
    
    def load_pretrained_encoder(self, encoder_state_dict):
        own_state = self.state_dict()
        for name, param in encoder_state_dict.items():
            if name.startswith('input_projection') or name.startswith('input_bn') or name.startswith('transformer_encoder'):
                if name in own_state:
                    own_state[name].copy_(param)


    def freeze_encoder(self):
        for name, param in self.named_parameters():
            if name.startswith('input_projection') or name.startswith('input_bn') or name.startswith('transformer_encoder'):
                param.requires_grad = False