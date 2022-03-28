import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, pre, act, dropout=0.):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.pre = pre
        self.act = act
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_dim), requires_grad=True)

    def forward(self, features):
        support = torch.spmm(features, self.weight)
        output = torch.spmm(self.pre, support)
        if self.act == 'relu':
            output = F.relu(output)
        output = self.dropout(output)

        return output
