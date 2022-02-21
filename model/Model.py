import torch
import torch.nn as nn
from model.GCN import GCN
from model.mlp import MLP


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, A_A, A_B):
        super().__init__()

        self.A_A = A_A
        self.A_B = A_B

        self.feature_idx_A = [i for i in range(A_A.shape[0])]
        self.features_A = torch.from_numpy(self.A_A[self.feature_idx_A])
        self.GNN_A = GCN(input_dim, output_dim, A_A)

        self.feature_idx_B = [i for i in range(A_B.shape[0])]
        self.features_B = torch.from_numpy(self.A_B[self.feature_idx_B])
        self.GNN_B = GCN(input_dim, output_dim, A_B)

        self.user_W_Attention_A_A = nn.Parameter(
            torch.randn(A_A.shape[0], input_dim),
            requires_grad=True
        )
        self.user_W_Attention_B_B = nn.Parameter(
            torch.randn(A_B.shape[0], input_dim),
            requires_grad=True
        )

        self.umlp_A = MLP(input_dim)
        self.imlp_A = MLP(input_dim)
        self.umlp_B = MLP(input_dim)
        self.imlp_B = MLP(input_dim)

    def forward(self, u, i, domain):
        features_A = self.GNN_A(self.features_A)

        features_B = self.GNN_A(self.features_B)

        user_A = features_A[u]
        user_W_Attention_A_A = self.user_W_Attention_A_A[u]
        user_W_Attention_B_A = 1 - user_W_Attention_A_A
        user_B = features_B[u]
        user_W_Attention_B_B = self.user_W_Attention_B_B[u]
        user_W_Attention_A_B = 1 - user_W_Attention_B_B

        if domain == 'A':
            item = features_A[i]
            user = torch.add(
                torch.mul(user_A, user_W_Attention_A_A),
                torch.mul(user_B, user_W_Attention_B_A)
            )
            user = self.umlp_A(user)
            item = self.imlp_A(item)
        else:
            item = features_B[i]
            user = torch.add(
                torch.mul(user_B, user_W_Attention_B_B),
                torch.mul(user_A, user_W_Attention_A_B)
            )
            user = self.umlp_B(user)
            item = self.imlp_B(item)

        return user, item
