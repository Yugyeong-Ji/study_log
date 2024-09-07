# Define the neural network
from classifier.model.modules import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class FSI_Transformer2(nn.Module):
    def __init__(self, args):
        self.args = args
        super(FSI_Transformer2, self).__init__()
        
        # Original input size and number of additional columns (alpha)
        self.original_input_size = args.input_size
        self.alpha = args.alpha  # Number of additional columns to add
        
        # Dimension for each feature
        self.dimension = args.dimension  # Dimension for each column
        
        # Initial weights
        self.initial_weight = nn.Parameter(torch.ones(1, self.args.input_size))
        
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dimension))
        
        # MLP to generate new columns (additional features)
        self.new_columns_generator = MLP(input_dim=args.input_size, hidden_dim=256, output_dim=self.alpha + args.input_size)
        
        self.new_column_bn = nn.BatchNorm1d(self.alpha + self.original_input_size)
        # Project the input (including new columns) to the desired dimension using MLP
        self.input_projection = MLP(input_dim=args.input_size + self.alpha, hidden_dim=256, output_dim=self.dimension)
        self.input_ln = nn.LayerNorm(self.dimension)
        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.dimension, args.input_size + self.alpha + 1)  # +1 for the cls token
        
        # Transformer encoder with multiple layers
        self.transformer_encoder = nn.Sequential(
            *[TransformerLayer(input_dimension=self.dimension) for _ in range(args.blocks)]
        )
        
        self.head_bn = nn.BatchNorm1d(self.dimension)
        
        self.transformer_decoder = nn.Sequential(
            *[TransformerLayer(input_dimension=self.dimension) for _ in range(4)]
        )
        
        # Classification head using only the classification token
        self.classification_head = MLP(input_dim=self.dimension, hidden_dim=128, output_dim=args.num_classes)

        self.classifier_bn = nn.BatchNorm1d(args.num_classes)
        
    def forward(self, x):
        # Input x shape: (batch_size, input_size)
        
        x = x * self.initial_weight
        
        # Generate new columns using the MLP
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

        
        # Only use the classification tokens for contrastive learning
        x = x[:, 0,:]  # Shape: (batch_size, dimension)

        x = self.head_bn(x)
        
        x= F.normalize(x, p=2, dim=1) 



        # Classification head
        x = self.classification_head(x)  # Shape: (batch_size, num_classes)
        
        
        x = self.classifier_bn(x)
        return x
    
    def load_pretrained_encoder(self, encoder_state_dict):
        for name, param in self.named_parameters():
            # We only load the weights for the layers before the classification head
            if name in encoder_state_dict and name.startswith(('new_columns_generator', 'input_projection', 'new_column_bn', 'input_ln', 'transformer_encoder', 'head_bn', 'cls_token', 'positional_encoding')):
                param.data.copy_(encoder_state_dict[name].data)
                if self.args.freeze:
                    param.requires_grad = False