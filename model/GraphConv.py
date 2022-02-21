import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, pre, dropout=0.):
        super().__init__()

        self.dropout = dropout
        self.pre = pre
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_dim), requires_grad=True)

    def forward(self, features):
        aggregate = torch.mm(self.pre, features)
        propagate = torch.mm(aggregate, self.weight) + self.bias
        out = F.relu(propagate)

        return out
