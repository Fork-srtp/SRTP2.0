from model.datareader import Datareader
from nltk import word_tokenize
from nltk.corpus import stopwords
import math
ca = []

class elem:
    def __init__(self, t, c):
        self.type=t
        # 1: word
        # 2: doc
        self.content=c

    def __str__(self):
        return "type: {}, content: {}".format(self.type,self.content)

def getPi(i:list):
    pass

def getPij(i:list,j:list):
    pass

def getPMI(i:list,j:list):
    res = math.log(getPij(i,j)/(getPi(i)*getPi(j)),10)
    return res

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