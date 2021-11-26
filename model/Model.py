import torch
import torch.nn as nn
from model.GCN import GCN
from model.mlp import MLP


class Model(nn.Module):
	def __init__(self, input_dim, output_dim, A, doc2vec):
		super().__init__()

		self.feature_idx = [i for i in range(A.shape[0])]
		self.doc2vec = doc2vec
		self.GNN = GCN(input_dim, output_dim, A)

		self.umlp = MLP(input_dim)
		self.imlp = MLP(input_dim)

	def forward(self, u, i):
		features = torch.from_numpy(self.doc2vec.dv.vectors[self.feature_idx])
		features = self.GNN(features)

		user = features[u]
		item = features[i]

		user = self.umlp(user)
		item = self.imlp(item)

		return user, item