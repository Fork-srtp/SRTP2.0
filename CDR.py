import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
import numpy as np
import argparse
import heapq
import math
import scipy.io as scio
from model.Model import Model
from dataset.datareader import Datareader
from dataset.Dataset import Dataset
import matplotlib.pyplot as plt

allResults = []

def main(dataName_A, dataName_B):
	parser = argparse.ArgumentParser(description="Options")
	parser.add_argument('-dataName_A',
						action='store',
						dest='dataName_A',
						default=dataName_A)
	parser.add_argument('-dataName_B',
						action='store',
						dest='dataName_B',
						default=dataName_B)
	parser.add_argument('-maxEpochs',
						action='store',
						dest='maxEpochs',
						default=100)
	parser.add_argument('-lr',
						action='store',
						dest='lr',
						default=0.001)
	parser.add_argument('-lambdad',
						action='store',
						dest='lambdad',
						default=0.001)
	parser.add_argument('-batchSize',
						action='store',
						dest='batchSize',
						default=4096)
	parser.add_argument('-negNum',
						action='store',
						dest='negNum',
						default=4)
	args = parser.parse_args()

	classifier = trainer(args)

	classifier.run()


class trainer:
	def __init__(self, args):
		self.dataName_A = args.dataName_A
		self.dataName_B = args.dataName_B
		self.maxEpochs = args.maxEpochs
		self.lr = args.lr
		self.batchSize = args.batchSize
		self.lambdad = args.lambdad
		self.negNum = args.negNum

		dr = Datareader(self.dataName_A, self.dataName_B)
		A_user_rating_dict, A_user_review_dict, A_item_user_dict, \
            B_user_rating_dict, B_user_review_dict, B_item_user_dict \
            	= dr.read_data()

		self.dataset = Dataset(
			A_user_rating_dict, A_user_review_dict, A_item_user_dict)
		self.adj, self.model_D2V = self.dataset.adj, self.dataset.model_D2V
		self.dataset.getTrainTest()
		self.dataset.getTrainDict()
		self.train, self.test = self.dataset.train, self.dataset.test
		self.testNeg = self.dataset.getTestNeg(self.test, self.negNum)

		self.maxRate = self.dataset.maxRate
		self.input_dim = self.model_D2V.vector_size
		self.output_dim = self.model_D2V.vector_size

	def run(self):

		model = Model(self.input_dim, self.output_dim, self.adj, self.model_D2V)
		optimizer = optim.Adam(model.parameters(), lr=self.lr)
		best_HR = -1
		best_NDCG = -1
		best_epoch = -1

		model.train()
		topK = 4
		for epoch in range(self.maxEpochs):
			print("=" * 20 + "Epoch ", epoch + 1, "=" * 20)
			train_u, train_i, train_r = self.dataset.getInstances(
				self.train, self.negNum)
			train_len = len(train_u)
			shuffle_idx = np.random.permutation(np.arange(train_len))
			train_u = train_u[shuffle_idx]
			train_i = train_i[shuffle_idx]
			train_r = train_r[shuffle_idx]

			num_batches = train_len // self.batchSize + 1

			loss = torch.zeros(0)
			for i in range(num_batches):
				min_idx = i * self.batchSize
				max_idx = np.min([train_len, (i + 1) * self.batchSize])

				if min_idx < train_len:
					train_u_batch = train_u[min_idx:max_idx]
					train_i_batch = train_i[min_idx:max_idx]
					train_r_batch = train_r[min_idx:max_idx]

					train_u_batch = torch.tensor(train_u_batch)
					train_i_batch = torch.tensor(train_i_batch)
					train_r_batch = torch.tensor(train_r_batch)

					user_out, item_out = model(train_u_batch, train_i_batch)

					norm_user_output = torch.sqrt(
						torch.sum(torch.square(user_out), axis=1))
					norm_item_output = torch.sqrt(
						torch.sum(torch.square(item_out), axis=1))
					regularizer = F.mse_loss(user_out, torch.zeros_like(user_out)) + F.mse_loss(
						item_out, torch.zeros_like(item_out))
					y_hat = torch.sum(
						torch.mul(user_out, item_out), axis=1
					) / (norm_user_output * norm_item_output)
					y_hat = y_hat + (1e-6) * (y_hat < 1e-6)
					regRate = train_r_batch / self.maxRate
					batchLoss = -(regRate * torch.log(
						y_hat) + (1 - regRate) * torch.log(
						1 - y_hat)) + self.lambdad * regularizer

					optimizer.zero_grad()
					batchLoss.backward(train_u_batch)
					optimizer.step()


					loss = torch.cat((loss, batchLoss.view(1,-1)), dim=1)

			loss = torch.mean(loss)
			print("\nMean Loss in epoch {} is: {}\n".format(epoch + 1, loss))

			model.eval()

			HR = []
			NDCG = []
			testUser = self.testNeg[0]
			testItem = self.testNeg[1]
			for i in range(len(testUser)):
				target = testItem[i][0]
				user_out, item_out = model(testUser[i], testItem[i])
				norm_user_output = torch.sqrt(
					torch.sum(torch.square(user_out), axis=1))
				norm_item_output = torch.sqrt(
					torch.sum(torch.square(item_out), axis=1))
				y_hat = torch.sum(
					torch.mul(user_out, item_out), axis=1
					) / (norm_user_output * norm_item_output)
				y_hat = y_hat + (1e-6) * (y_hat < 1e-6)

				item_score_dict = {}

				for j in range(len(testItem[i])):
					item = testItem[i][j]
					item_score_dict[item] = y_hat[j]

				ranklist = heapq.nlargest(topK,
										  item_score_dict,
										  key=item_score_dict.get)

				tmp_HR = 0
				for item in ranklist:
					if item == target:
						tmp_HR = 1

				HR.append(tmp_HR)

				tmp_NDCG = 0
				for i in range(len(ranklist)):
					item = ranklist[i]
					if item == target:
						tmp_NDCG = math.log(2) / math.log(i + 2)

				NDCG.append(tmp_NDCG)

			HR = np.mean(HR)
			NDCG = np.mean(NDCG)
			allResults.append([epoch + 1, topK, HR, NDCG, loss.detach().numpy()])
			print(
				"Epoch ", epoch + 1,
				"TopK: {} HR: {}, NDCG: {}".format(
					topK, HR, NDCG))

			if HR > best_HR:
				best_HR = HR
				best_epoch = epoch + 1
			if NDCG > best_NDCG:
				best_NDCG = NDCG
				best_epoch = epoch + 1

		print(
			"Best HR: {}, NDCG: {} At Epoch {}".format(best_HR, best_NDCG, best_epoch))

		bestPerformance = [[best_HR, best_NDCG, best_epoch]]
		matname = 'baseline_result.mat'
		scio.savemat(matname, {
						'allResults': allResults,
						'bestPerformance': bestPerformance
					})

if __name__ == '__main__':
	main('Arts_Crafts_and_Sewing_5', 'Luxury_Beauty_5')
	allResults = np.array(allResults)
	x_label = allResults[:,0]
	y_topK = allResults[:,1]
	y_HR = allResults[:,2]
	y_NDCG = allResults[:,3]
	y_loss = allResults[:,4]

	fig1 = plt.figure()
	f1 = fig1.add_subplot(1, 1, 1)
	f1.set_title('Loss')
	f1.set_xlabel('Epoch')
	f1.set_ylabel('Loss')
	f1.grid()
	f1.plot(x_label, y_loss)
	fig1.savefig('Loss.jpg')

	fig2 = plt.figure()
	f2 = fig2.add_subplot(1, 1, 1)
	f2.set_title('HR')
	f2.set_xlabel('Epoch')
	f2.set_ylabel('HR')
	f2.grid()
	f2.plot(x_label, y_HR)
	fig2.savefig('HR.jpg')

	fig3 = plt.figure()
	f3 = fig3.add_subplot(1, 1, 1)
	f2.set_title('NDCG')
	f3.set_xlabel('Epoch')
	f3.set_ylabel('NDCG')
	f3.grid()
	f3.plot(x_label, y_NDCG)
	fig3.savefig('NDCG.jpg')

	plt.show()

    # plt.plot()

