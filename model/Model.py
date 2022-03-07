import torch
import torch.nn as nn
from model.GCN import GCN
from model.mlp import MLP
from model.ReviewGCN import ReviewGCN
import pytorch_lightning as pl


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Model(nn.Module):
    def __init__(self, output_dim, Rating_A, Rating_B, Review_A, Review_B):
        super().__init__()

        self.A_A = Rating_A
        self.A_B = Rating_B

        self.rating_idx_A = [i for i in range(Rating_A.shape[0])]
        # self.features_A = torch.from_numpy(self.A_A[self.rating_idx_A])
        self.rating_A = self.A_A[self.rating_idx_A]
        self.GNN_A = GCN(Rating_A)
        self.ReviewGCN_A = ReviewGCN(Review_A)
        self.review_A = torch.ones_like(Review_A)

        self.rating_idx_B = [i for i in range(Rating_B.shape[0])]
        # self.features_B = torch.from_numpy(self.A_B[self.rating_idx_B])
        self.rating_B = self.A_B[self.rating_idx_B]
        self.GNN_B = GCN(Rating_B)
        self.ReviewGCN_B = ReviewGCN(Review_B)
        self.review_B = torch.ones_like(Review_B)

        self.user_W_Attention_A_A = nn.Parameter(
            torch.randn(Rating_A.shape[0], output_dim),
            requires_grad=True
        )
        self.user_W_Attention_B_B = nn.Parameter(
            torch.randn(Rating_B.shape[0], output_dim),
            requires_grad=True
        )

        self.umlp_A = MLP(output_dim)
        self.imlp_A = MLP(output_dim)
        self.umlp_B = MLP(output_dim)
        self.imlp_B = MLP(output_dim)

    def forward(self, u, i, domain):
        rating_A = self.GNN_A(self.rating_A)
        review_A = self.ReviewGCN_A(self.review_A)
        user_A = rating_A[u] + review_A[u]

        rating_B = self.GNN_B(self.rating_B)
        review_B = self.ReviewGCN_B(self.review_B)
        user_B = rating_B[u] + review_B[u]

        user_W_Attention_A_A = self.user_W_Attention_A_A[u]
        user_W_Attention_B_A = 1 - user_W_Attention_A_A

        user_W_Attention_B_B = self.user_W_Attention_B_B[u]
        user_W_Attention_A_B = 1 - user_W_Attention_B_B

        if domain == 'A':
            item = rating_A[i]
            user = torch.add(
                torch.mul(user_A, user_W_Attention_A_A),
                torch.mul(user_B, user_W_Attention_B_A)
            )
            user = self.umlp_A(user)
            item = self.imlp_A(item)
        else:
            item = rating_B[i]
            user = torch.add(
                torch.mul(user_B, user_W_Attention_B_B),
                torch.mul(user_A, user_W_Attention_A_B)
            )
            user = self.umlp_B(user)
            item = self.imlp_B(item)

        return user, item



class Model_pl(pl.LightningModule):
    def __init__(self, output_dim, Rating_A, Rating_B, Review_A, Review_B):
        super().__init__()

        self.A_A = Rating_A
        self.A_B = Rating_B

        self.rating_idx_A = [i for i in range(Rating_A.shape[0])]
        # self.features_A = torch.from_numpy(self.A_A[self.rating_idx_A])
        self.rating_A = self.A_A[self.rating_idx_A]
        print('rating_A: ', self.rating_A)
        self.GNN_A = GCN(Rating_A)
        print('GNN_A: ', self.GNN_A)
        self.ReviewGCN_A = ReviewGCN(Review_A)
        self.review_A = torch.ones_like(Review_A)

        self.rating_idx_B = [i for i in range(Rating_B.shape[0])]
        # self.features_B = torch.from_numpy(self.A_B[self.rating_idx_B])
        self.rating_B = self.A_B[self.rating_idx_B]
        print('rating_B: ', self.rating_B)
        self.GNN_B = GCN(Rating_B)
        print('GNN_B: ', self.GNN_B)
        self.ReviewGCN_B = ReviewGCN(Review_B)
        self.review_B = torch.ones_like(Review_B)

        self.user_W_Attention_A_A = nn.Parameter(
            torch.randn(Rating_A.shape[0], output_dim),
            requires_grad=True
        )
        self.user_W_Attention_B_B = nn.Parameter(
            torch.randn(Rating_B.shape[0], output_dim),
            requires_grad=True
        )

        self.umlp_A = MLP(output_dim)
        self.imlp_A = MLP(output_dim)
        self.umlp_B = MLP(output_dim)
        self.imlp_B = MLP(output_dim)

    def forward(self, u, i, domain):
        rating_A = self.GNN_A(self.rating_A)
        review_A = self.ReviewGCN_A(self.review_A)
        user_A = rating_A[u] + review_A[u]

        rating_B = self.GNN_B(self.rating_B)
        review_B = self.ReviewGCN_B(self.review_B)
        user_B = rating_B[u] + review_B[u]

        user_W_Attention_A_A = self.user_W_Attention_A_A[u]
        user_W_Attention_B_A = 1 - user_W_Attention_A_A

        user_W_Attention_B_B = self.user_W_Attention_B_B[u]
        user_W_Attention_A_B = 1 - user_W_Attention_B_B

        if domain == 'A':
            item = rating_A[i]
            user = torch.add(
                torch.mul(user_A, user_W_Attention_A_A),
                torch.mul(user_B, user_W_Attention_B_A)
            )
            user = self.umlp_A(user)
            item = self.imlp_A(item)
        else:
            item = rating_B[i]
            user = torch.add(
                torch.mul(user_B, user_W_Attention_B_B),
                torch.mul(user_A, user_W_Attention_A_B)
            )
            user = self.umlp_B(user)
            item = self.imlp_B(item)

        return user, item

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
