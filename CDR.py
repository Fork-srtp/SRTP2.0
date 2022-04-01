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
from model.Discriminator import Discriminator
from dataset.datareader import Datareader
from dataset.Dataset import Dataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model.ReviewGraph import GraphBuilder
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import preprocess_adj
allResults = []



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    parser.add_argument('-lr_G',
                        action='store',
                        dest='lr_G',
                        default=0.001)
    parser.add_argument('-lr_D',
                        action='store',
                        dest='lr_D',
                        default=0.0000)                    
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
                        default=7)
    parser.add_argument('-rating_dim',
                        action='store',
                        dest='rating_dim',
                        default=200)
    parser.add_argument('-review_dim',
                        action='store',
                        dest='review_dim',
                        default=20)
    parser.add_argument('-emb_dim',
                        action='store',
                        dest='emb_dim',
                        default=200)
    parser.add_argument('-topK',
                        action='store',
                        dest='topK',
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
        self.lr_G = args.lr_G
        self.lr_D = args.lr_D
        self.batchSize = args.batchSize
        self.lambdad = args.lambdad
        self.negNum = args.negNum
        self.rating_dim = args.rating_dim
        self.review_dim = args.review_dim
        self.emb_dim = args.emb_dim
        self.topK = args.topK

        dr = Datareader(self.dataName_A, self.dataName_B)
        A_user_rating_dict, A_user_review_dict, A_item_user_dict, \
        B_user_rating_dict, B_user_review_dict, B_item_user_dict \
            = dr.read_data()

        self.dataset_A = Dataset(
            A_user_rating_dict, A_user_review_dict, A_item_user_dict)
        self.dataset_B = Dataset(
            B_user_rating_dict, B_user_review_dict, B_item_user_dict)
        self.ReviewGraph = GraphBuilder(A_user_review_dict,
                               A_item_user_dict,
                               B_user_review_dict,
                               B_item_user_dict)

        self.Review_A, self.Review_B = self.ReviewGraph.adj()
        self.Review_A = preprocess_adj(self.Review_A).to(device)
        self.Review_B = preprocess_adj(self.Review_B).to(device)

        self.adj_A = self.dataset_A.adj
        self.adj_A =preprocess_adj(self.adj_A).to(device)

        self.dataset_A.getTrainTest()
        self.dataset_A.getTrainDict()
        self.train_A, self.test_A = self.dataset_A.train, self.dataset_A.test
        self.testNeg_A = self.dataset_A.getTestNeg(self.test_A, 99)

        self.adj_B = self.dataset_B.adj
        self.adj_B = preprocess_adj(self.adj_B).to(device)

        self.dataset_B.getTrainTest()
        self.dataset_B.getTrainDict()
        self.train_B, self.test_B = self.dataset_B.train, self.dataset_B.test
        self.testNeg_B = self.dataset_B.getTestNeg(self.test_B, 99)

        self.maxRate_A = self.dataset_A.maxRate
        self.maxRate_B = self.dataset_B.maxRate


    def run(self):

        model = Model(self.rating_dim, self.review_dim, self.adj_A, self.adj_B, self.Review_A, self.Review_B)
        d = Discriminator(self.review_dim)
        d = d.to(device)
        model = model.to(device)
        writer = SummaryWriter('runs/latest')
        print('training on device:', device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        optimizer_G_params = [model.ReviewGCN_A.layer1.weight,
                              model.ReviewGCN_A.layer1.bias,
                              model.ReviewGCN_A.layer2.weight,
                              model.ReviewGCN_A.layer2.bias,
                              model.ReviewGCN_B.layer1.weight,
                              model.ReviewGCN_B.layer1.bias,
                              model.ReviewGCN_B.layer2.weight,
                              model.ReviewGCN_B.layer2.bias]
        optimizer_G = optim.Adam(optimizer_G_params, lr=self.lr_G)
        optimizer_D = optim.Adam(d.parameters(), lr=self.lr_D)
        best_HR_A = -1
        best_NDCG_A = -1
        best_epoch_A = -1
        best_HR_B = -1
        best_NDCG_B = -1
        best_epoch_B = -1

        model.train()
        topK = self.topK
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

            loss_A = torch.zeros(0).to(device)
            for i in range(num_batches):
                min_idx = i * self.batchSize
                max_idx = np.min([train_len_A, (i + 1) * self.batchSize])

                if min_idx < train_len_A:
                    train_u_batch = train_u_A[min_idx:max_idx]
                    train_i_batch = train_i_A[min_idx:max_idx]
                    train_r_batch = train_r_A[min_idx:max_idx]

                    train_u_batch = torch.tensor(train_u_batch).to(device)
                    train_i_batch = torch.tensor(train_i_batch).to(device)
                    train_r_batch = torch.tensor(train_r_batch).to(device)

                    user_out, item_out, Review_A, Review_B = model(train_u_batch, train_i_batch, 'A')

                    label_A = d(Review_A.detach())
                    label_B = d(Review_B.detach())
                    # print("label_A:{}".format(torch.mean(label_A)))
                    # print("label_B:{}".format(torch.mean(label_B)))

                    label_A = torch.maximum(torch.zeros_like(label_A) + 1e-6, label_A)
                    label_B = torch.maximum(torch.zeros_like(label_B) + 1e-6, label_B)

                    loss_d = torch.mean(- (torch.ones_like(label_A) * torch.log(
                        label_A) + (1 - torch.ones_like(label_A)) * torch.log(
                        1 - label_A)) - (torch.zeros_like(label_B) * torch.log(
                        label_B) + (1 - torch.zeros_like(label_B)) * torch.log(
                        1 - label_B)))

                    optimizer_D.zero_grad()
                    loss_d.backward()
                    optimizer_D.step()

                    label_A = d(Review_A)
                    label_B = d(Review_B)

                    label_A = torch.maximum(torch.zeros_like(label_A) + 1e-6, label_A)
                    label_B = torch.maximum(torch.zeros_like(label_B) + 1e-6, label_B)

                    loss_g = torch.mean((torch.ones_like(label_A) * torch.log(
                        label_A) + (1 - torch.ones_like(label_A)) * torch.log(
                        1 - label_A)) + (torch.zeros_like(label_B) * torch.log(
                        label_B) + (1 - torch.zeros_like(label_B)) * torch.log(
                        1 - label_B)))
                    optimizer_G.zero_grad()
                    loss_g.backward(retain_graph=True)

                    norm_user_output = torch.sqrt(
                        torch.sum(torch.square(user_out), axis=1))
                    norm_item_output = torch.sqrt(
                        torch.sum(torch.square(item_out), axis=1))
                    regularizer = F.mse_loss(user_out, torch.zeros_like(user_out), reduction='sum') + F.mse_loss(
                        item_out, torch.zeros_like(item_out), reduction='sum')
                    if torch.sum(norm_user_output * norm_item_output):
                        y_hat = torch.sum(
                            torch.mul(user_out, item_out), axis=1
                        ) / (norm_user_output * norm_item_output)
                    else:
                        y_hat = torch.sum(
                            torch.mul(user_out, item_out), axis=1
                        )
                    y_hat = torch.maximum(torch.zeros_like(y_hat) + 1e-6, y_hat)
                    regRate = train_r_batch / self.maxRate_A
                    batchLoss_A = torch.sum(-(regRate * torch.log(
                        y_hat) + (1 - regRate) * torch.log(
                        1 - y_hat))) + self.lambdad * regularizer

                    # print("batchLoss_A:{}".format(torch.mean(batchLoss_A)))

                    optimizer.zero_grad()
                    batchLoss_A.backward()
                    optimizer.step()
                    optimizer_G.step()
                    loss_A = torch.cat((loss_A, torch.tensor([batchLoss_A]).to(device)))

            loss_A = torch.mean(loss_A)
            print("Mean Loss_A in epoch {} is: {}\n".format(epoch + 1, loss_A))
            writer.add_scalar('loss/loss_A', loss_A, global_step=epoch)

            # domain B
            train_u_B, train_i_B, train_r_B = self.dataset_B.getInstances(
                self.train_B, self.negNum)
            train_len_B = len(train_u_B)
            shuffle_idx = np.random.permutation(np.arange(train_len_B))
            train_u_B = train_u_B[shuffle_idx]
            train_i_B = train_i_B[shuffle_idx]
            train_r_B = train_r_B[shuffle_idx]

            num_batches = train_len_B // self.batchSize + 1

            loss_B = torch.zeros(0).to(device)
            for i in range(num_batches):
                min_idx = i * self.batchSize
                max_idx = np.min([train_len_B, (i + 1) * self.batchSize])

                if min_idx < train_len_B:
                    train_u_batch = train_u_B[min_idx:max_idx]
                    train_i_batch = train_i_B[min_idx:max_idx]
                    train_r_batch = train_r_B[min_idx:max_idx]

                    train_u_batch = torch.tensor(train_u_batch).to(device)
                    train_i_batch = torch.tensor(train_i_batch).to(device)
                    train_r_batch = torch.tensor(train_r_batch).to(device)

                    user_out, item_out, Review_A, Review_B = model(train_u_batch, train_i_batch, 'B')

                    label_A = d(Review_A.detach())
                    label_B = d(Review_B.detach())

                    label_A = torch.maximum(torch.zeros_like(label_A) + 1e-6, label_A)
                    label_B = torch.maximum(torch.zeros_like(label_B) + 1e-6, label_B)

                    loss_d = torch.mean(- (torch.ones_like(label_A) * torch.log(
                        label_A) + (1 - torch.ones_like(label_A)) * torch.log(
                        1 - label_A)) - (torch.zeros_like(label_B) * torch.log(
                        label_B) + (1 - torch.zeros_like(label_B)) * torch.log(
                        1 - label_B)))

                    optimizer_D.zero_grad()
                    loss_d.backward()
                    optimizer_D.step()

                    label_A = d(Review_A)
                    label_B = d(Review_B)

                    label_A = torch.maximum(torch.zeros_like(label_A) + 1e-6, label_A)
                    label_B = torch.maximum(torch.zeros_like(label_B) + 1e-6, label_B)

                    loss_g = torch.mean((torch.ones_like(label_A) * torch.log(
                        label_A) + (1 - torch.ones_like(label_A)) * torch.log(
                        1 - label_A)) + (torch.zeros_like(label_B) * torch.log(
                        label_B) + (1 - torch.zeros_like(label_B)) * torch.log(
                        1 - label_B)))
                    # print(torch.mean(loss_g))
                    optimizer_G.zero_grad()
                    loss_g.backward(retain_graph=True)
                    # optimizer_G.step()

                    norm_user_output = torch.sqrt(
                        torch.sum(torch.square(user_out), axis=1))
                    norm_item_output = torch.sqrt(
                        torch.sum(torch.square(item_out), axis=1))
                    regularizer = F.mse_loss(user_out, torch.zeros_like(user_out), reduction='sum') + F.mse_loss(
                        item_out, torch.zeros_like(item_out), reduction='sum')
                    if torch.sum(norm_user_output * norm_item_output):
                        y_hat = torch.sum(
                            torch.mul(user_out, item_out), axis=1
                        ) / (norm_user_output * norm_item_output)
                    else:
                        y_hat = torch.sum(
                            torch.mul(user_out, item_out), axis=1
                        )
                    y_hat = torch.maximum(torch.zeros_like(y_hat) + 1e-6, y_hat)
                    regRate = train_r_batch / self.maxRate_B
                    batchLoss_B = torch.sum(-(regRate * torch.log(
                        y_hat) + (1 - regRate) * torch.log(
                        1 - y_hat))) + self.lambdad * regularizer

                    optimizer.zero_grad()
                    batchLoss_B.backward()
                    optimizer.step()
                    optimizer_G.step()

                    loss_B = torch.cat((loss_B, torch.tensor([batchLoss_B]).to(device)))

            loss_B = torch.mean(loss_B)
            print("Mean Loss_B in epoch {} is: {}\n".format(epoch + 1, loss_B))
            writer.add_scalar('loss/loss_B', loss_B, global_step=epoch)

            # print("loss_g in epoch {} is: {}\n".format(epoch + 1, torch.mean(loss_g)))

            # if (epoch + 1) % 5 != 0:
            #     continue
            model.eval()

            HR_A = []
            NDCG_A = []
            testUser_A = self.testNeg_A[0]
            testItem_A = self.testNeg_A[1]
            HR_B = []
            NDCG_B = []
            testUser_B = self.testNeg_B[0]
            testItem_B = self.testNeg_B[1]
            print('testUser : ', len(testUser_A))
            for i in tqdm(range(len(testUser_A))):
                # A
                target = testItem_A[i][0]
                user_out, item_out, _, _ = model(testUser_A[i], testItem_A[i], 'A')
                norm_user_output = torch.sqrt(
                    torch.sum(torch.square(user_out), axis=1))
                norm_item_output = torch.sqrt(
                    torch.sum(torch.square(item_out), axis=1))
                y_hat = torch.sum(
                    torch.mul(user_out, item_out), axis=1
                ) / (norm_user_output * norm_item_output)
                y_hat = torch.maximum(torch.zeros_like(y_hat) + 1e-6, y_hat)

                item_score_dict = {}

                for j in range(len(testItem_A[i])):
                    item = testItem_A[i][j]
                    item_score_dict[item] = y_hat[j]

                ranklist = heapq.nlargest(topK,
                                          item_score_dict,
                                          key=item_score_dict.get)

                tmp_HR = 0
                for item in ranklist:
                    if item == target:
                        tmp_HR = 1

                HR_A.append(tmp_HR)

                tmp_NDCG = 0
                for i in range(len(ranklist)):
                    item = ranklist[i]
                    if item == target:
                        tmp_NDCG = math.log(2) / math.log(i + 2)

                NDCG_A.append(tmp_NDCG)

                # B
                target = testItem_B[i][0]
                user_out, item_out, _, _ = model(testUser_B[i], testItem_B[i], 'B')
                norm_user_output = torch.sqrt(
                    torch.sum(torch.square(user_out), axis=1))
                norm_item_output = torch.sqrt(
                    torch.sum(torch.square(item_out), axis=1))
                y_hat = torch.sum(
                    torch.mul(user_out, item_out), axis=1
                ) / (norm_user_output * norm_item_output)
                y_hat = torch.maximum(torch.zeros_like(y_hat) + 1e-6, y_hat)

                item_score_dict = {}

                for j in range(len(testItem_B[i])):
                    item = testItem_B[i][j]
                    item_score_dict[item] = y_hat[j]

                ranklist = heapq.nlargest(topK,
                                          item_score_dict,
                                          key=item_score_dict.get)

                tmp_HR = 0
                for item in ranklist:
                    if item == target:
                        tmp_HR = 1

                HR_B.append(tmp_HR)

                tmp_NDCG = 0
                for i in range(len(ranklist)):
                    item = ranklist[i]
                    if item == target:
                        tmp_NDCG = math.log(2) / math.log(i + 2)

                NDCG_B.append(tmp_NDCG)

            HR_A = np.mean(HR_A)
            NDCG_A = np.mean(NDCG_A)
            HR_B = np.mean(HR_B)
            NDCG_B = np.mean(NDCG_B)

            writer.add_scalar('others/HR', HR_A, global_step=epoch)
            writer.add_scalar('others/NDCG', NDCG_A, global_step=epoch)

            allResults.append([epoch + 1, topK, HR_A, NDCG_A, loss_A.detach().cpu().numpy(), HR_B, NDCG_B, loss_B.detach().cpu().numpy()])
            print(
                "Domain A Epoch: ", epoch + 1,
                "TopK: {} HR: {}, NDCG: {}".format(
                    topK, HR_A, NDCG_A))
            print(
                "Domain B Epoch: ", epoch + 1,
                "TopK: {} HR: {}, NDCG: {}".format(
                    topK, HR_B, NDCG_B))


            if HR_A > best_HR_A:
                best_HR_A = HR_A
                best_epoch_A = epoch + 1
            if NDCG_A > best_NDCG_A:
                best_NDCG_A = NDCG_A
                best_epoch_A = epoch + 1

            if HR_B > best_HR_B:
                best_HR_B = HR_B
                best_epoch_B = epoch + 1
            if NDCG_B > best_NDCG_B:
                best_NDCG_B = NDCG_B
                best_epoch_B = epoch + 1

        print(
            "Domain A: Best HR: {}, NDCG: {} At Epoch {}".format(best_HR_A, best_NDCG_A, best_epoch_A))
        print(
            "Domain B: Best HR: {}, NDCG: {} Bt Epoch {}".format(best_HR_B, best_NDCG_B, best_epoch_B))

        bestPerformance = [[best_HR_A, best_NDCG_A, best_epoch_A],
                           [best_HR_B, best_NDCG_B, best_epoch_B]]
        model.eval()
        with torch.no_grad():
            userVecs = [model(u, testItem_A[0], 'A')[0] for u in testUser_A]
            itemVecs = [model(testUser_A[0], i, 'A')[1] for i in testItem_A]
            userVecs = [i.detach().cpu().numpy().reshape(-1) for i in userVecs]
            itemVecs = [i.detach().cpu().numpy().reshape(-1) for i in itemVecs]

            tsne = TSNE(n_components=2, init='pca', random_state=0)
            user2D = tsne.fit_transform(userVecs)
            item2D = tsne.fit_transform(itemVecs)

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
    main('Appliances', 'Movies_and_TV')
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