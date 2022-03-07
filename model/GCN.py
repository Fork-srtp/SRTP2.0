import torch
import torch.nn as nn
from model.GraphConv import GraphConv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GCN(nn.Module):
    def __init__(self, A):
        super().__init__()

        self.I = torch.eye(A.shape[0]).to(device)
        self.A_hat = A + self.I
        self.D = torch.sum(self.A_hat, axis=0)
        self.D = torch.sqrt(self.D)
        self.D = torch.diag(self.D)
        self.D_inv = torch.inverse(self.D)
        self.pre = torch.mm(torch.mm(self.D_inv, self.A_hat), self.D_inv)

        self.layer1 = GraphConv(A.shape[0], 200, self.pre, dropout=0.5)
        self.layer2 = GraphConv(200, 20, self.pre, dropout=0.5)

    def forward(self, features):
        x = self.layer1(features)
        # dropout
        x = self.layer2(x)

        return x
