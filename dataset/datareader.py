from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import random

class Datareader:
    def __init__(self, dataName_A, dataName_B):
        # source domain
        self.A_user_rating_dict = {}
        self.A_user_review_dict = {}
        self.A_item_user_dict = {}

        # target domain
        self.B_user_rating_dict = {}
        self.B_user_review_dict = {}
        self.B_item_user_dict = {}

        self.dataName_A = dataName_A
        self.dataName_B = dataName_B

    def read_data(self):
        print("Loading Domain-A dataset...")
        A_user_dict = {}
        filePath_A = 'data/' + self.dataName_A + '.json'
        with open(filePath_A, 'r') as f:
            for line in f.readlines():
                lines = json.loads(line)
                user = lines['reviewerID']
                item = lines['asin']
                if 'summary' not in lines.keys():
                    continue
                review = lines['summary']
                rating = float(lines['overall'])
                if user not in A_user_dict:
                    A_user_dict[user] = []
                A_user_dict[user] += [[item, review, rating]]

        print("Loading Domain-B dataset...")
        B_user_dict = {}
        filePath_B = 'data/' + self.dataName_B + '.json'
        with open(filePath_B, 'r') as f:
            for line in f.readlines():
                lines = json.loads(line)
                user = lines['reviewerID']
                item = lines['asin']
                if 'summary' not in lines.keys():
                    continue
                review = lines['summary']
                rating = float(lines['overall'])
                if user not in B_user_dict:
                    B_user_dict[user] = []
                B_user_dict[user] += [[item, review, rating]]

        # data filter

        # filter inactive users
        temp_keys = list(A_user_dict.keys())
        for user in tqdm(temp_keys):
            if len(A_user_dict[user]) < 5:
                A_user_dict.pop(user)

        # filter uncommon users
        temp_keys = list(A_user_dict.keys())
        for user in tqdm(temp_keys):
            if user not in B_user_dict:
                A_user_dict.pop(user)

        temp_keys = list(B_user_dict.keys())
        for user in tqdm(temp_keys):
            if user not in A_user_dict:
                B_user_dict.pop(user)

        for user in tqdm(A_user_dict.keys()):
            for each in A_user_dict[user]:
                item = each[0]
                review = each[1]
                rating = each[2]
                if user not in self.A_user_rating_dict:
                    self.A_user_rating_dict[user] = []
                    self.A_user_review_dict[user] = []
                self.A_user_review_dict[user] += [[item, review]]
                self.A_user_rating_dict[user] += [[item, rating]]


        for user in tqdm(B_user_dict.keys()):
            for each in B_user_dict[user]:
                item = each[0]
                review = each[1]
                rating = each[2]
                if user not in self.B_user_rating_dict:
                    self.B_user_rating_dict[user] = []
                    self.B_user_review_dict[user] = []
                self.B_user_review_dict[user] += [[item, review]]
                self.B_user_rating_dict[user] += [[item, rating]]

        # construct item dictionary

        for user in tqdm(A_user_dict.keys()):
            for each in A_user_dict[user]:
                if each[0] not in self.A_item_user_dict:
                    self.A_item_user_dict[each[0]] = []
                self.A_item_user_dict[each[0]] += [[user, each[1]]]

        for user in tqdm(B_user_dict.keys()):
            for each in B_user_dict[user]:
                if each[0] not in self.B_item_user_dict:
                    self.B_item_user_dict[each[0]] = []
                self.B_item_user_dict[each[0]] += [[user, each[1]]]

        return self.A_user_rating_dict, self.A_user_review_dict, self.A_item_user_dict, \
            self.B_user_rating_dict, self.B_user_review_dict, self.B_item_user_dict
