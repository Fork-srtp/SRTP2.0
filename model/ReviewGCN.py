import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GraphConv import GraphConv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReviewGCN(nn.Module):
    def __init__(self, emb_dim, A):
        super().__init__()

        self.pre = A

        self.layer1 = GraphConv(A.shape[0], 2 * emb_dim, self.pre, 'relu', dropout=0.5)
        self.layer2 = GraphConv(2 * emb_dim, emb_dim, self.pre, None, dropout=0.5)

    def forward(self, features):
        x = self.layer1(features)
        # dropout
        x = self.layer2(x)

        return x
