import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1), :]
        return x
