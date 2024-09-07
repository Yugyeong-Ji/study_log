import torch.nn as nn
from .mlp import MLP

class TransformerLayer(nn.Module):
    
    def __init__(self, input_dimension: int = 128, nhead: int = 8,  hidden_dim: int = 512, dropout: float = 0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dimension, nhead, dropout=dropout)
        self.mlp_encode = MLP(input_dimension, hidden_dim, input_dimension * 3)
        self.mlp = MLP(input_dimension, hidden_dim, input_dimension)

        self.norm1 = nn.LayerNorm(input_dimension)
        self.norm2 = nn.LayerNorm(input_dimension)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # Apply MLP to the input tensor and reshape
        x_encoded = self.mlp_encode(x) #  x : (batch, input_size, 512)
        # print(f"x_encoded shape :{x_encoded.shape}")
        batch_size, seq_len, _ = x_encoded.size()
        
        # Reshape and split the result into three parts along the feature dimension
        x_encoded = x_encoded.view(batch_size, seq_len, 3, -1)
        q, k, v = x_encoded.chunk(3, dim=2)
        
        # Reshape q, k, v to the expected shape by self_attn (L, N, E)
        q = q.view(batch_size, seq_len, -1).transpose(0, 1)  # (L, N, E)
        k = k.view(batch_size, seq_len, -1).transpose(0, 1)  # (L, N, E)
        v = v.view(batch_size, seq_len, -1).transpose(0, 1)  # (L, N, E)

        # Self-attention mechanism
        attn_output, _ = self.self_attn(q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        
        # Residual connection and normalization
        attn_output = attn_output.transpose(0, 1)  # (N, L, E)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feedforward network with custom MLP
        x2 = self.mlp(x)
        x2 = self.dropout2(x2)
        
        # Residual connection and normalization
        x = x + x2
        x = self.norm2(x)
        return x