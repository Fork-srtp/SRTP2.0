from nltk import word_tokenize
from nltk.corpus import stopwords
import torch
import numpy as np
import networkx as nx
import itertools
import math
from collections import defaultdict
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from utils import print_graph_detail



def get_window(content_lst, window_size):
    """
    找出窗口
    :param content_lst:
    :param window_size:
    :return:
    """
    word_window_freq = defaultdict(int)  # w(i)  单词在窗口单位内出现的次数
    word_pair_count = defaultdict(int)  # w(i, j)
    windows_len = 0
    for words in tqdm(content_lst, desc="Split by window"):
        windows = list()

        if isinstance(words, str):
            words = words.split()
        length = len(words)

        if length <= window_size:
            windows.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(list(set(window)))

        for window in windows:
            for word in window:
                word_window_freq[word] += 1

            for word_pair in itertools.combinations(window, 2):
                word_pair_count[word_pair] += 1

        windows_len += len(windows)
    return word_window_freq, word_pair_count, windows_len


def cal_pmi(W_ij, W, word_freq_1, word_freq_2):
    p_i = word_freq_1 / W
    p_j = word_freq_2 / W
    p_i_j = W_ij / W
    pmi = math.log(p_i_j / (p_i * p_j))

    return pmi


def count_pmi(windows_len, word_pair_count, word_window_freq, threshold):
    word_pmi_lst = list()
    for word_pair, W_i_j in tqdm(word_pair_count.items(), desc="Calculate pmi between words"):
        word_freq_1 = word_window_freq[word_pair[0]]
        word_freq_2 = word_window_freq[word_pair[1]]

        pmi = cal_pmi(W_i_j, windows_len, word_freq_1, word_freq_2)
        if pmi <= threshold:
            continue
        word_pmi_lst.append([word_pair[0], word_pair[1], pmi])
    return word_pmi_lst


def get_pmi_edge(content_lst, window_size=20, threshold=0.):
    pmi_start = time()
    word_window_freq, word_pair_count, windows_len = get_window(content_lst,
                                                                window_size=window_size)

    pmi_edge_lst = count_pmi(windows_len, word_pair_count, word_window_freq, threshold)
    print("Total number of edges between word:", len(pmi_edge_lst))
    pmi_time = time() - pmi_start
    return pmi_edge_lst, pmi_time

class GraphBuilder:
    def __init__(self,
                 A_user_review_dict,
                 A_item_user_dict,
                 B_user_review_dict,
                 B_item_user_dict):
        self.gA = nx.Graph()
        self.gB = nx.Graph()
        self.word2id_A = dict()
        self.word2id_B = dict()

        self.interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#',
                                  '$','%']  # 定义标点符号列表
        self.stops = set(stopwords.words("english"))

        self.review_A, self.review_B = self.getReview(A_user_review_dict,
                                                 A_item_user_dict,
                                                 B_user_review_dict,
                                                 B_item_user_dict)
        self.get_tfidf_edge()
        self.get_pmi_edge()

    def adj(self):
        e = list(self.gA.edges.data())
        node_num = self.gA.number_of_nodes()
        adj_A = torch.eye(node_num, node_num)

        for edge in e:
            adj_A[edge[0]][edge[1]] = edge[2]['weight']
            adj_A[edge[1]][edge[0]] = edge[2]['weight']
        adj_A = adj_A.type(torch.FloatTensor)

        e = list(self.gB.edges.data())
        node_num = self.gB.number_of_nodes()
        adj_B = torch.eye(node_num, node_num)

        for edge in e:
            adj_B[edge[0]][edge[1]] = edge[2]['weight']
            adj_B[edge[1]][edge[0]] = edge[2]['weight']
        adj_B = adj_B.type(torch.FloatTensor)

        return adj_A, adj_B

    def getWords(self, sentence):
        cutwords1 = word_tokenize(sentence)
        cutwords2 = [word for word in cutwords1 if word not in self.interpunctuations]  # 去除标点符号
        # cutwords3 = [word for word in cutwords2 if word not in self.stops]
        tmp = ""
        for word in cutwords2:
            tmp += word + " "
        tmp = tmp.lower()
        return tmp

    def getReview(self,
                 A_user_review_dict,
                 A_item_user_dict,
                 B_user_review_dict,
                 B_item_user_dict
                 ):
        # Domain A review mat
        print("Get Review_A Graph Begining...")
        review_A = list()

        # get 某一个user的所有review
        for user, reviews in A_user_review_dict.items():
            str = ""
            for meta_review in reviews:
                str += " "
                str += meta_review[1]
            str = self.getWords(str)
            review_A.append(str)

        # get 某一个item的所有review
        for item, reviews in A_item_user_dict.items():
            str = ""
            for meta_review in reviews:
                str += " "
                str += meta_review[1]
            str = self.getWords(str)
            review_A.append(str)

        print("A done...")

        # Domain B review mat
        print("Get Review_B Graph Begining...")
        review_B = list()

        # get 某一个user的所有review
        for user, reviews in B_user_review_dict.items():
            str = ""
            for meta_review in reviews:
                str += " "
                str += meta_review[1]
            str = self.getWords(str)
            review_B.append(str)

        # get 某一个item的所有review
        for item, reviews in B_item_user_dict.items():
            str = ""
            for meta_review in reviews:
                str += " "
                str += meta_review[1]
            str = self.getWords(str)
            review_B.append(str)

        print("B done...")

        return review_A, review_B

    def get_pmi_edge(self):
        # A
        pmi_edge_lst, pmi_time = get_pmi_edge(self.review_A, window_size=20, threshold=0.0)
        print("pmi time A:", pmi_time)

        for edge_item in pmi_edge_lst:
            word_indx1 = self.node_num_A + self.word2id_A[edge_item[0]]
            word_indx2 = self.node_num_A + self.word2id_A[edge_item[1]]
            if word_indx1 == word_indx2:
                continue
            self.gA.add_edge(word_indx1, word_indx2, weight=edge_item[2])

        print_graph_detail(self.gA)

        # B
        pmi_edge_lst, pmi_time = get_pmi_edge(self.review_B, window_size=20, threshold=0.0)
        print("pmi time B:", pmi_time)

        for edge_item in pmi_edge_lst:
            word_indx1 = self.node_num_B + self.word2id_B[edge_item[0]]
            word_indx2 = self.node_num_B + self.word2id_B[edge_item[1]]
            if word_indx1 == word_indx2:
                continue
            self.gB.add_edge(word_indx1, word_indx2, weight=edge_item[2])

        print_graph_detail(self.gB)

    def get_tfidf_edge(self):
        # 获得tfidf权重矩阵（sparse）和单词列表
        tfidf_vec_A, tfidf_vec_B = self.get_tfidf_vec()

        # A
        count_lst_A = list()  # 统计每个句子的长度
        for ind, row in tqdm(enumerate(tfidf_vec_A),
                             desc="generate tfidf edge"):
            count = 0
            for col_ind, value in zip(row.indices, row.data):
                word_ind = self.node_num_A + col_ind
                self.gA.add_edge(ind, word_ind, weight=value)
                count += 1
            count_lst_A.append(count)

        # B
        count_lst_B = list()  # 统计每个句子的长度
        for ind, row in tqdm(enumerate(tfidf_vec_B),
                             desc="generate tfidf edge"):
            count = 0
            for col_ind, value in zip(row.indices, row.data):
                word_ind = self.node_num_B + col_ind
                self.gB.add_edge(ind, word_ind, weight=value)
                count += 1
            count_lst_B.append(count)

    def get_tfidf_vec(self):
        """
        学习获得tfidf矩阵，及其对应的单词序列
        :param content_lst:
        :return:
        """

        # Domain A
        text_tfidf_A = Pipeline([
            ("vect", CountVectorizer(min_df=1,
                                     max_df=1.0,
                                     token_pattern=r"\S+",
                                     )),
            ("tfidf", TfidfTransformer(norm=None,
                                       use_idf=True,
                                       smooth_idf=False,
                                       sublinear_tf=False
                                       ))
        ])

        # tfidf_vec_A = text_tfidf_A.fit_transform(np.array(self.review_A))
        tfidf_vec_A = text_tfidf_A.fit_transform(self.review_A)

        self.node_num_A = tfidf_vec_A.shape[0]

        # 映射单词
        vocab_lst_A = text_tfidf_A["vect"].get_feature_names_out()
        print("vocab_lst len:", len(vocab_lst_A))
        for ind, word in enumerate(vocab_lst_A):
            self.word2id_A[word] = ind

        self.vocab_lst_A = vocab_lst_A

        # Domain B
        text_tfidf_B = Pipeline([
            ("vect", CountVectorizer(min_df=1,
                                     max_df=1.0,
                                     token_pattern=r"\S+",
                                     )),
            ("tfidf", TfidfTransformer(norm=None,
                                       use_idf=True,
                                       smooth_idf=False,
                                       sublinear_tf=False
                                       ))
        ])

        # tfidf_vec_B = text_tfidf_B.fit_transform(np.array(self.review_B))
        tfidf_vec_B = text_tfidf_B.fit_transform(self.review_B)

        self.node_num_B = tfidf_vec_B.shape[0]
        self.gB.add_nodes_from([i for i in range(tfidf_vec_B.shape[0] + tfidf_vec_B.shape[1])])

        # 映射单词
        vocab_lst_B = text_tfidf_B["vect"].get_feature_names_out()
        print("vocab_lst len:", len(vocab_lst_B))
        for ind, word in enumerate(vocab_lst_B):
            self.word2id_B[word] = ind

        self.vocab_lst_B = vocab_lst_B

        return tfidf_vec_A, tfidf_vec_B