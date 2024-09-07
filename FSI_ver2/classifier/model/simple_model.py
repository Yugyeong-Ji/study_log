# Define the neural network
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, args):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(args.input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, args.num_classes)



    def forward(self, x):
        
        print(f"input shape : {x.shape}")
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # print(f"x shape : {x.shape}")
        
        return x