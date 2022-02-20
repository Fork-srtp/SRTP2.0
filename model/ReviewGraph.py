from model.datareader import Datareader
from nltk import word_tokenize
from nltk.corpus import stopwords
import math
ca = []
size = 3 # sliding windows size

class elem:
    def __init__(self, t, c):
        self.type=t
        # 1: word
        # 2: doc
        self.content=c

    def __str__(self):
        return "type: {}, content: {}".format(self.type,self.content)


def getElement(i,j):
    if ca[i].content == ca[j].content:
        return 1
    elif ca[i].type == 1 and ca[j].type == 1:
        return getPMI(ca[i].content,ca[j].content)
    elif ca[i].type == 1 and ca[j].type == 2:
        pass
    else:
        return 0

def getPi(stringi:list):
    w = len(ca) + 1 - size
    w_i = 0
    for i in range(0,len(ca)-size):
        isI = False
        for j in range(i,i+size):
            if(ca[j].content == stringi):
                isI = True
        if isI:
            w_i += 1
    return w_i/w


def getPij(stringi:list,stringj:list):
    w = len(ca) + 1 - size
    w_ij  = 0
    for i in range(0,len(ca)-size):
        isI = False
        isJ = False
        for j in range(i,i+size):
            if(ca[j].content == stringi):
                isI = True
            if(ca[j].content == stringj):
                isJ = True
        if(isI and isJ):
            w_ij += 1
    return w_ij/w


def getPMI(stringi:list,stringj:list):
    res = math.log(getPij(stringi,stringj)/(getPi(stringi)*getPi(stringj)),10)
    return res




def TF_IDF(doc: int, word: int):
    doc_str = ca[doc].content
    word_str = ca[word].content
    doc_list = word_tokenize(doc_str)
    tf = doc_list.count(word_str) / len(doc_list)
    idf = math.log(getCntByType(2) / (IDF_cnt(word_str) + 1), 10)
    return tf * idf

def getCntByType(type:int):
    cnt = 0
    for i in range(0,len(ca)):
        if ca[i].type == type:
            cnt += 1
    return cnt


def IDF_cnt(content:str):
    cnt = 0
    for i in range(0,len(ca)):
        if ca[i].type == 2:
            if content in ca[i].content:
                cnt += 1
    return cnt


if __name__ == '__main__':
    dr = Datareader('Arts_Crafts_and_Sewing_5', 'Luxury_Beauty_5')
    A_user_rating_dict, A_user_review_dict, A_item_user_dict, \
        B_user_rating_dict, B_user_review_dict, B_item_user_dict = dr.read_data()


    full_review = ''
    # get 某一个user的所有review
    user_review ={}
    for user_name in A_user_review_dict:
        reviews = A_user_review_dict[user_name]
        review_cat = ''
        for meta_review in reviews:
            review_cat += meta_review[1]
        user_review[user_name] = review_cat
        full_review += review_cat
        # print(user_name, review_cat)




    # get 某一个item的所有review
    item_review ={}
    for item_name in A_item_user_dict:
        reviews = A_item_user_dict[item_name]
        review_cat = ''
        for meta_review in reviews:
            review_cat += meta_review[1]
        item_review[item_name] = review_cat
        full_review += review_cat

    cutwords1 = word_tokenize(full_review)
    interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']  # 定义标点符号列表
    cutwords2 = [word for word in cutwords1 if word not in interpunctuations]  # 去除标点符号
    print(cutwords2)


    stops = set(stopwords.words("english"))
    cutwords3 = [word for word in cutwords2 if word not in stops]
    print(cutwords3)
    ca = [elem(1, word) for word in cutwords3]
    print(len(ca))
    ca.extend([elem(2, user_review[doc]) for doc in user_review])
    print(len(ca))
    ca.extend([elem(2, item_review[doc]) for doc in item_review])
    print(len(ca))
    # word2idx = {word: idx for word, idx in enumerate(cutwords3)}



    a=2