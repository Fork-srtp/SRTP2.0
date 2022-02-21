from model.datareader import Datareader
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
import math
import numpy as np





class elem:
    def __init__(self, t, c):
        self.type=t
        # 1: word
        # 2: doc
        self.content=c

    def __str__(self):
        return "type: {}, content: {}".format(self.type,self.content)

# 分词




class GraphBuilder:
    def __init__(self):
        self.window_size = 10  # sliding windows size
        self.ca = []
        self.windows = []
        self.word_window_freq = {}
        self.word_pair_count = {}
        self.word_id_map = {}

    def getWords(self,doc: list):

        cutwords0 = sent_tokenize(doc)
        cutwords1 = [word_tokenize(sentence) for sentence in cutwords0]
        interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$',
                             '%']  # 定义标点符号列表
        cutwords2 = [word for sentence in cutwords1 for word in sentence if word not in interpunctuations]  # 去除标点符号
        stops = set(stopwords.words("english"))
        cutwords3 = [word for word in cutwords2 if word not in stops]
        cutwords3 = list(set(cutwords3))
        return cutwords3


    def getWord2IdMap(self):
        for i in range(len(self.ca)):
            if self.ca[i].type == 1:
                self.word_id_map[self.ca[i].content] = i

    def computeWindows(self):
        self.getWord2IdMap()  # get map
        print('word_id_map', len(self.word_id_map))

        for i in range(len(self.ca)):
            if self.ca[i].type == 2:
                words = self.getWords(self.ca[i].content)
                length = len(words)
                if length <= self.window_size:
                    self.windows.append(words)
                else:
                    for j in range(length - self.window_size + 1):
                        window = words[j: j + self.window_size]
                        self.windows.append(window)
        print(self.windows)
        for window in self.windows:
            appeared = set()
            for i in range(len(window)):
                if window[i] in appeared:
                    continue
                if window[i] in self.word_window_freq:
                    self.word_window_freq[window[i]] += 1
                else:
                    self.word_window_freq[window[i]] = 1
                appeared.add(window[i])
        for i in self.ca:
            if i.content == 'shorelines':
                print('shorelines!')
            else:
                pass
                # print(i.content)
        for i in range(len(self.word_id_map)):
            for j in range(len(self.word_id_map)):
                str1 = str(i)+','+str(j)
                self.word_pair_count[str1] = 0
        for window in self.windows:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = self.word_id_map[word_i]
                    word_j = window[j]
                    word_j_id = self.word_id_map[word_j]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    # if word_pair_str in self.word_pair_count:
                    self.word_pair_count[word_pair_str] += 1
                    # else:
                    #     self.word_pair_count[word_pair_str] = 1
                    # two orders
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    # if word_pair_str in self.word_pair_count:
                    self.word_pair_count[word_pair_str] += 1
                    # else:
                    #     self.word_pair_count[word_pair_str] = 1
        print('word_id_map', len(self.word_id_map))


    def getElement(self,i, j):
        if self.ca[i].content == self.ca[j].content:
            # print(1)
            return 1

        elif self.ca[i].type == 1 and self.ca[j].type == 1:
            # print(2)
            return self.getPMI(self.ca[i].content, self.ca[j].content)
        elif self.ca[i].type == 2 and self.ca[j].type == 1:
            # print(3)
            return self.TF_IDF(i, j)
        else:
            # print(4)
            return 0

    def getPi(self,stringi: list):
        windows_num = len(self.windows)
        return self.word_window_freq[stringi] / windows_num

    def getPij(self,stringi: list, stringj: list):
        windows_num = len(self.windows)
        str1 = str(self.word_id_map[stringi]) + ',' + str(self.word_id_map[stringj])
        return self.word_pair_count[str1] / windows_num

    def getPMI(self,stringi: list, stringj: list):
        # res = math.log(getPij(stringi,stringj,ca)/(getPi(stringi,ca)*getPi(stringj,ca)),10)
        r0 = self.getPij(stringi, stringj)
        r1 = self.getPi(stringi)
        r2 = self.getPi(stringj)
        if r1 != 0 and r2 != 0 and r0 > 0:
            return math.log(r0 / (r1 * r2), 10)
        else:
            return 0

    def getCntByType(self,type: int):
        cnt = 0
        for i in range(0, len(self.ca)):
            if self.ca[i].type == type:
                cnt += 1
        return cnt

    def IDF_cnt(self,content: str):
        cnt = 0
        for i in range(0, len(self.ca)):
            if self.ca[i].type == 2:
                if content in self.ca[i].content:
                    cnt += 1
        return cnt

    def TF_IDF(self,doc: int, word: int):
        doc_str = self.ca[doc].content
        word_str = self.ca[word].content
        doc_list = word_tokenize(doc_str)
        tf = doc_list.count(word_str) / len(doc_list)
        idf = math.log(self.getCntByType(2) / (self.IDF_cnt(word_str) + 1), 10)
        return tf * idf
    # if __name__ == '__main__':
    def getGraph(self,
                 A_user_review_dict,
                 A_item_user_dict,
                 B_user_review_dict,
                 B_item_user_dict
                 ):
        # for user_name in B_user_review_dict:
        #     if user_name in A_user_review_dict:
        #         A_user_review_dict[user_name].extend(B_user_review_dict[user_name])
        #     else:
        #         A_user_review_dict[user_name] = B_user_review_dict[user_name]

        # for item_name in B_item_user_dict:
        #     if item_name in A_item_user_dict:
        #         A_item_user_dict[item_name].extend(B_item_user_dict)
        #     else:
        #         A_item_user_dict[item_name] = B_item_user_dict[item_name]




        # A

        print("Get Review_A Graph Begining...")

        full_review = ''
        # get 某一个user的所有review
        user_review = {}
        A_user_review_dict.update()
        for user_name in A_user_review_dict:
            reviews = A_user_review_dict[user_name]
            review_cat = ' '
            for meta_review in reviews:
                review_cat += meta_review[1]
            user_review[user_name] = review_cat
            full_review += review_cat
            # print(user_name, review_cat)

        # get 某一个item的所有review
        item_review = {}
        for item_name in A_item_user_dict:
            reviews = A_item_user_dict[item_name]
            review_cat = ' '
            for meta_review in reviews:
                review_cat += meta_review[1]
            item_review[item_name] = review_cat
            full_review += review_cat
        print('full: ', full_review)
        cutwords0 = sent_tokenize(full_review)
        cutwords1 = [word_tokenize(sentence) for sentence in cutwords0]
        interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$',
                             '%']  # 定义标点符号列表
        cutwords2 = [word for sentence in cutwords1 for word in sentence if word not in interpunctuations]  # 去除标点符号

        stops = set(stopwords.words("english"))
        cutwords3 = [word for word in cutwords2 if word not in stops]
        cutwords3 = list(set(cutwords3))
        self.ca = [elem(1, word) for word in cutwords3]
        print(len(self.ca))
        self.ca.extend([elem(2, user_review[doc]) for doc in user_review])
        self.ca.extend([elem(2, item_review[doc]) for doc in item_review])
        print(len(self.ca))
        # word2idx = {word: idx for word, idx in enumerate(cutwords3)}
        self.computeWindows()
        Review_A = np.ones([len(self.ca), len(self.ca)], np.float32)

        for i in range(len(self.ca)):
            for j in range(len(self.ca)):
                # print(i, j, len(self.ca))
                Review_A[i][j] = self.getElement(i, j)

        print("A done...")

        # B
        print("Get Review_B Graph Begining...")
        self.ca = []
        full_review = ''
        # get 某一个user的所有review
        user_review = {}
        B_user_review_dict.update()
        for user_name in B_user_review_dict:
            reviews = B_user_review_dict[user_name]
            review_cat = ' '
            for meta_review in reviews:
                review_cat += meta_review[1]
            user_review[user_name] = review_cat
            full_review += review_cat
            # print(user_name, review_cat)

        # get 某一个item的所有review
        item_review = {}
        for item_name in B_item_user_dict:
            reviews = B_item_user_dict[item_name]
            review_cat = ' '
            for meta_review in reviews:
                review_cat += meta_review[1]
            item_review[item_name] = review_cat
            full_review += review_cat

        cutwords0 = sent_tokenize(full_review)
        cutwords1 = [word_tokenize(sentence) for sentence in cutwords0]
        interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$',
                             '%']  # 定义标点符号列表
        cutwords2 = [word for sentence in cutwords1 for word in sentence if word not in interpunctuations]  # 去除标点符号

        stops = set(stopwords.words("english"))
        cutwords3 = [word for word in cutwords2 if word not in stops]
        cutwords3 = list(set(cutwords3))

        # print(cutwords3)
        self.ca = [elem(1, word) for word in cutwords3]
        # print(len(ca))
        self.ca.extend([elem(2, user_review[doc]) for doc in user_review])
        # print(len(ca))
        self.ca.extend([elem(2, item_review[doc]) for doc in item_review])
        # print(len(ca))
        # word2idx = {word: idx for word, idx in enumerate(cutwords3)}
        self.computeWindows()
        Review_B = np.ones([len(self.ca), len(self.ca)], np.float32)
        for i in range(len(self.ca)):
            for j in range(len(self.ca)):
                Review_B[i][j] = self.getElement(i, j)

        print("B done...")

        return Review_A, Review_B