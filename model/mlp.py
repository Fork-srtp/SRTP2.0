import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim * 2)
        self.act1 = nn.Tanh()

        self.fc2 = nn.Linear(input_dim * 2, input_dim * 4)
        self.act2 = nn.Tanh()

        self.fc3 = nn.Linear(input_dim * 4, input_dim * 8)
        self.act3 = nn.Tanh()

        self.fc4 = nn.Linear(input_dim * 8, input_dim * 4)
        self.act4 = nn.Tanh()

        self.fc5 = nn.Linear(input_dim * 4, input_dim * 2)
        self.act5 = nn.Tanh()

        self.fc6 = nn.Linear(input_dim * 2, input_dim)
        self.act6 = nn.Tanh()

    def forward(self, input):
        x = input

        x = self.fc1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.act2(x)

        x = self.fc3(x)
        x = self.act3(x)

        x = self.fc4(x)
        x = self.act4(x)

        x = self.fc5(x)
        x = self.act5(x)

        x = self.fc6(x)
        x = self.act6(x)
        return x
