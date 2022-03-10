import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, emb_dim):
        super(Discriminator, self).__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(emb_dim, 2 * emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(2 * emb_dim, 2 * emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(2 * emb_dim, 2 * emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(2 * emb_dim, 1),
        #     nn.Sigmoid()
        # )
        self.norm = nn.BatchNorm2d(emb_dim)
        self.fc1 = nn.Linear(emb_dim, 2 * emb_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(2 * emb_dim, 2 * emb_dim)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(2 * emb_dim, 2 * emb_dim)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(2 * emb_dim, 1)
        self.act4 = nn.Sigmoid()

    def forward(self, emb):
        # label = self.net(emb)
        label = self.fc1(emb)
        label = self.act1(label)
        label = self.fc2(label)
        label = self.act2(label)
        label = self.fc3(label)
        label = self.act3(label)
        label = self.fc4(label)
        label = self.act4(label)
        return label.view(-1)