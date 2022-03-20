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
        aggregate = torch.mm(self.pre, features)
        propagate = torch.mm(aggregate, self.weight) + self.bias
        if self.act == 'relu':
            out = F.relu(propagate)
        else:
            out = propagate
        out = self.dropout(out)

        return out
