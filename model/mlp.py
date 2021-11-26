import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim * 2)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(input_dim * 2, input_dim * 4)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(input_dim * 4, input_dim * 8)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(input_dim * 8, input_dim * 4)
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(input_dim * 4, input_dim * 2)
        self.relu5 = nn.ReLU()

        self.fc6 = nn.Linear(input_dim * 2, input_dim)
        self.relu6 = nn.ReLU()

    def forward(self, input):
        x = input

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        x = self.relu4(x)

        x = self.fc5(x)
        x = self.relu5(x)
        
        x = self.fc6(x)
        x = self.relu6(x)
        return x
