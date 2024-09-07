# Define the neural network
from .modules import *
import torch
import torch.nn as nn

class TabularCLIP(nn.Module):
    def __init__(self, args):
        super(TabularCLIP, self).__init__()
        
        # Input size and dimension
        self.args = args
        self.input_size = args.input_size
        self.dimension = args.dimension
        self.alpha = args.alpha
        
        # Initial weights
        self.initial_weight = nn.Parameter(torch.ones(1, self.input_size))
        
        # MLP to generate new columns (additional features)
        self.new_columns_generator = MLP(input_dim=self.input_size, hidden_dim=256, output_dim=self.alpha + self.input_size)
        
        self.new_column_bn = nn.BatchNorm1d(self.alpha + self.input_size)
        

        # MLP to project the input (including new columns) to the desired dimension
        self.input_projection = MLP(input_dim=self.input_size + self.alpha, hidden_dim=256, output_dim=self.dimension)
        
        self.input_ln = nn.LayerNorm(self.dimension)
        
        # Classification token for each input
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dimension))
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.dimension, self.input_size + self.alpha + 1)  # +1 for the cls token
        
        # Batch normalization for the projected input

        
        # Transformer encoder with multiple layers
        self.transformer_encoder = nn.Sequential(
            *[TransformerLayer(input_dimension=self.dimension) for _ in range(args.blocks)]
        )
        
        self.head_bn = nn.BatchNorm1d(self.dimension)

        
    def forward(self, x):
        
        x = x * self.initial_weight
        
        x = self.new_columns_generator(x)  # Shape: (batch_size, alpha + input_size)
        
        x = self.new_column_bn(x)
        # Project input (including new columns) to (batch_size, input_size + alpha, 128)
        x = self.input_projection(x)  # Shape: (batch_size, input_size + alpha, 128)
        x = self.input_ln(x)
        
        # Append the classification token to each input
        batch_size = x.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, dimension)
        x = torch.cat((cls_token, x.unsqueeze(1)), dim=1)  # Shape: (batch_size, input_size + alpha + 1, dimension)

        # Positional Encoding
        x = self.positional_encoding(x)

        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # Shape: (batch_size, input_size + alpha + 1, dimension)

        """ classification Token 정보를 뽑을지, 아니면 다른 token을 뽑을지"""
        # if self.args.test :
        #     tokens = x[:, 1,:]# Shape : (batch_size, input_size + alpha, dimension)
        #     x = x[:, 0,:]  # Shape: (batch_size, dimension)

        #     return tokens, self.head_bn(x)
        
        # else :
        
        # Only use the classification tokens for contrastive learning
        x = x[:, 0,:]  # Shape: (batch_size, dimension)
        
        x =  self.head_bn(x)
        
        return x



    def load_pretrained_encoder(self, encoder_state_dict):
        for name, param in self.named_parameters():
            # We only load the weights for the layers before the classification head
            if name in encoder_state_dict and name.startswith(('new_columns_generator', 'input_projection', 'new_column_bn', 'input_ln', 'transformer_encoder', 'head_bn', 'cls_token', 'positional_encoding')):
                param.data.copy_(encoder_state_dict[name].data)
