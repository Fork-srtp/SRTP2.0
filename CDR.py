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
from sklearn.manifold import TSNE
from model.ReviewGraph import GraphBuilder
from model.ReviewGraphBuilder import BuildGraph
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
                        default=1)
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
    parser.add_argument('-input_dim',
                        action='store',
                        dest='input_dim',
                        default=10)
    parser.add_argument('-output_dim',
                        action='store',
                        dest='output_dim',
                        default=10)
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
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim

        dr = Datareader(self.dataName_A, self.dataName_B)
        A_user_rating_dict, A_user_review_dict, A_item_user_dict, \
        B_user_rating_dict, B_user_review_dict, B_item_user_dict \
            = dr.read_data()

        self.dataset_A = Dataset(
            A_user_rating_dict, A_user_review_dict, A_item_user_dict)
        self.dataset_B = Dataset(
            B_user_rating_dict, B_user_review_dict, B_item_user_dict)
        # self.jj = GraphBuilder()
        self.jj = BuildGraph()
        self.Review_A, self.Review_B = self.jj.getGraph(A_user_review_dict,
                                                A_item_user_dict,
                                                B_user_review_dict,
                                                B_item_user_dict)

        self.adj_A = self.dataset_A.adj
        self.dataset_A.getTrainTest()
        self.dataset_A.getTrainDict()
        self.train_A, self.test_A = self.dataset_A.train, self.dataset_A.test
        self.testNeg_A = self.dataset_A.getTestNeg(self.test_A, self.negNum)

        self.adj_B = self.dataset_B.adj
        self.dataset_B.getTrainTest()
        self.dataset_B.getTrainDict()
        self.train_B, self.test_B = self.dataset_B.train, self.dataset_B.test
        self.testNeg_B = self.dataset_B.getTestNeg(self.test_B, self.negNum)

        self.maxRate_A = self.dataset_A.maxRate
        self.maxRate_B = self.dataset_B.maxRate

    def run(self):

        model = Model(self.input_dim, self.output_dim, self.adj_A, self.adj_B, self.Review_A, self.Review_B)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        best_HR = -1
        best_NDCG = -1
        best_epoch = -1

        model.train()
        topK = 4
        for epoch in range(self.maxEpochs):
            print("=" * 20 + "Epoch ", epoch + 1, "=" * 20)
            train_u_A, train_i_A, train_r_A = self.dataset_A.getInstances(
                self.train_A, self.negNum)
            train_len_A = len(train_u_A)
            shuffle_idx = np.random.permutation(np.arange(train_len_A))
            train_u_A = train_u_A[shuffle_idx]
            train_i_A = train_i_A[shuffle_idx]
            train_r_A = train_r_A[shuffle_idx]

            num_batches = train_len_A // self.batchSize + 1

            loss_A = torch.zeros(0)
            for i in range(num_batches):
                min_idx = i * self.batchSize
                max_idx = np.min([train_len_A, (i + 1) * self.batchSize])

                if min_idx < train_len_A:
                    train_u_batch = train_u_A[min_idx:max_idx]
                    train_i_batch = train_i_A[min_idx:max_idx]
                    train_r_batch = train_r_A[min_idx:max_idx]

                    train_u_batch = torch.tensor(train_u_batch)
                    train_i_batch = torch.tensor(train_i_batch)
                    train_r_batch = torch.tensor(train_r_batch)

                    user_out, item_out = model(train_u_batch, train_i_batch, 'A')

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
                    regRate = train_r_batch / self.maxRate_A
                    batchLoss_A = -(regRate * torch.log(
                        y_hat) + (1 - regRate) * torch.log(
                        1 - y_hat)) + self.lambdad * regularizer

                    optimizer.zero_grad()
                    batchLoss_A.backward(train_u_batch)
                    optimizer.step()

                    loss_A = torch.cat((loss_A, batchLoss_A.view(1, -1)), dim=1)

            loss_A = torch.mean(loss_A)
            print("\nMean Loss_A in epoch {} is: {}\n".format(epoch + 1, loss_A))

            # domain B
            train_u_B, train_i_B, train_r_B = self.dataset_B.getInstances(
                self.train_B, self.negNum)
            train_len_B = len(train_u_B)
            shuffle_idx = np.random.permutation(np.arange(train_len_B))
            train_u_B = train_u_B[shuffle_idx]
            train_i_B = train_i_B[shuffle_idx]
            train_r_B = train_r_B[shuffle_idx]

            num_batches = train_len_B // self.batchSize + 1

            loss_B = torch.zeros(0)
            for i in range(num_batches):
                min_idx = i * self.batchSize
                max_idx = np.min([train_len_B, (i + 1) * self.batchSize])

                if min_idx < train_len_B:
                    train_u_batch = train_u_B[min_idx:max_idx]
                    train_i_batch = train_i_B[min_idx:max_idx]
                    train_r_batch = train_r_B[min_idx:max_idx]

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
                    regRate = train_r_batch / self.maxRate_B
                    batchLoss_B = -(regRate * torch.log(
                        y_hat) + (1 - regRate) * torch.log(
                        1 - y_hat)) + self.lambdad * regularizer

                    optimizer.zero_grad()
                    batchLoss_B.backward(train_u_batch)
                    optimizer.step()

                    loss_B = torch.cat((loss_B, batchLoss_B.view(1, -1)), dim=1)

            loss_B = torch.mean(loss_B)
            print("\nMean Loss_A in epoch {} is: {}\n".format(epoch + 1, loss_B))



            model.eval()

            HR = []
            NDCG = []
            testUser = self.testNeg_A[0]
            testItem = self.testNeg_A[1]
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
        model.eval()
        with torch.no_grad():
            userVecs = [model(u, testItem[0])[0] for u in testUser]
            itemVecs = [model(testUser[0], i)[1] for i in testItem]
            userVecs = [i.detach().numpy().reshape(-1) for i in userVecs]
            itemVecs = [i.detach().numpy().reshape(-1) for i in itemVecs]

            tsne = TSNE(n_components=2, init='pca', random_state=0)
            user2D = tsne.fit_transform(userVecs)
            item2D = tsne.fit_transform(itemVecs)

            # for idx, user in enumerate(testUser):
            # 	user_out, item_out = model(testUser[idx], testItem[idx])
            # 	userVecs.append(user_out)

            fig0 = plt.figure()
            t1 = plt.scatter(user2D[:, 0], user2D[:, 1], marker='x', c='r', s=20)  # marker:点符号 c:点颜色 s:点大小
            t2 = plt.scatter(item2D[:, 0], item2D[:, 1], marker='o', c='b', s=20)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend((t1, t2), ('user', 'item'))
            plt.show()

        matname = 'baseline_result.mat'
        scio.savemat(matname, {
            'allResults': allResults,
            'bestPerformance': bestPerformance
        })


if __name__ == '__main__':
    main('Arts_Crafts_and_Sewing_5', 'Luxury_Beauty_5')
    allResults = np.array(allResults)
    x_label = allResults[:, 0]
    y_topK = allResults[:, 1]
    y_HR = allResults[:, 2]
    y_NDCG = allResults[:, 3]
    y_loss = allResults[:, 4]

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
