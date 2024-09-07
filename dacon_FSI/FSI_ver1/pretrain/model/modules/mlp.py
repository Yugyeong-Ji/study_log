import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Adding dropout between layers
            nn.Linear(hidden_dim, output_dim),
            # nn.Dropout(0.1)  # Adding dropout between layers
        )
        
    def forward(self, x):
        return self.layers(x)