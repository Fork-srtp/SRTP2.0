import numpy as np
import torch
import networkx as nx
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

from dataset.Dataset import Dataset
from dataset.datareader import Datareader
from utils import print_graph_detail


if __name__ == '__main__':
    dr = Datareader('Arts_Crafts_and_Sewing_5', 'Luxury_Beauty_5')
    A_user_rating_dict, A_user_review_dict, A_item_user_dict, \
    B_user_rating_dict, B_user_review_dict, B_item_user_dict \
        = dr.read_data()

    dataset_A = Dataset(
        A_user_rating_dict, A_user_review_dict, A_item_user_dict)
    dataset_B = Dataset(
        B_user_rating_dict, B_user_review_dict, B_item_user_dict)

    dataset_A.writeGraph('A_user_item_graph.txt')
    dataset_B.writeGraph('B_user_item_graph.txt')
