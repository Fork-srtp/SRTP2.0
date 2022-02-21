import torch
import torch.nn as nn
from model.GraphConv import GraphConv


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, A):
        super().__init__()

        I = torch.eye(A.shape[0])
        A_hat = A + I
        D = torch.sum(A_hat, axis=0)
        D = torch.sqrt(D)
        D = torch.diag(D)
        D_inv = torch.inverse(D)
        pre = torch.mm(torch.mm(D_inv, A_hat), D_inv)

        self.layer1 = GraphConv(input_dim, 16, pre, dropout=0.5)
        self.layer2 = GraphConv(16, output_dim, pre, dropout=0.5)

    def forward(self, features):
        x = self.layer1(features)
        x = self.layer2(x)

        return x
