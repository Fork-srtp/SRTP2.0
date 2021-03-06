import numpy as np
import torch
import networkx as nx
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from utils import print_graph_detail
samplling_probability = 0.05

class Dataset(object):
	def __init__(self, user_rating_dict, user_review_dict, item_user_dict):
		self.graph = nx.Graph()
		self.data, self.adj = self.getData(
			user_rating_dict, user_review_dict, item_user_dict)
		self.train, self.test = self.getTrainTest()
		self.trainDict = self.getTrainDict()


	def getData(self, user_rating_dict, user_review_dict, item_user_dict):
		maxr = 0.0
		data = []

		nodeset = [_key for _key in user_review_dict.keys()]
		nodeset += [_key for _key in item_user_dict.keys()]

		nodedict = {}

		for index, node in enumerate(nodeset):
			nodedict[node] = index

		u = 0
		for user, reviews in user_review_dict.items():
			u += 1
			str = ""
			for each in reviews:
				str += " "
				str += each[1]

		i = 0
		for item, reviews in item_user_dict.items():
			i += 1
			str = ""
			for each in reviews:
				str += " "
				str += each[1]

		self.shape = [u, i]
		print("user: {} item: {}".format(u, i))
		self.graph.add_nodes_from(range(u+i))

		for user, rating in user_rating_dict.items():
			for each in rating:
				data.append((nodedict[user], nodedict[each[0]], each[1]))
				if each[1] > maxr:
					maxr = each[1]
				# self.graph.add_weighted_edges_from([(nodedict[user], nodedict[each[0]], each[1])])

		self.maxRate = maxr

		data = sorted(data, key=lambda x: x[0])
		for i in range(len(data)-1):
			user = data[i][0]
			item = data[i][1]
			rate = data[i][2]
			if data[i][0] != data[i+1][0]:
				pass
			else:
				self.graph.add_weighted_edges_from([(user, item, rate / maxr)])

		# for i in range(self.graph.number_of_nodes()):
			# self.graph.add_weighted_edges_from([(i, i, 1)])
		print_graph_detail(self.graph)

		return data, self.graph

	def getTrainTest(self):
		data = self.data
		data = sorted(data, key=lambda x: x[0])
		train = []
		test = []
		for i in range(len(data)-1):
			user = data[i][0]
			item = data[i][1]
			rate = data[i][2]
			if data[i][0] != data[i+1][0]:
				test.append((user, item, rate))
			else:
				train.append((user, item, rate))

		test.append((data[-1][0]-1, data[-1][1]-1, data[1][2]))
		return train, test

	def getTrainDict(self):
		dataDict = {}
		for i in self.train:
			dataDict[(i[0], i[1])] = i[2]
		return dataDict

	def getInstances(self, data, negNum):
		user = []
		item = []
		rate = []
		for i in data:
			user.append(i[0])
			item.append(i[1])
			rate.append(i[2])
			for t in range(negNum):
				j = np.random.randint(self.shape[1]) + self.shape[0]
				while (i[0], j) in self.trainDict:
					j = np.random.randint(self.shape[1]) + self.shape[0]
				user.append(i[0])
				item.append(j)
				rate.append(0.0)
		return np.array(user), np.array(item), np.array(rate)

	def getTestNeg(self, testData, negNum):
		user = []
		item = []
		for s in testData:
			tmp_user = []
			tmp_item = []
			u = s[0]
			i = s[1]
			tmp_user.append(u)
			tmp_item.append(i)
			neglist = set()
			neglist.add(i)
			for t in range(negNum):
				j = np.random.randint(self.shape[1]) + self.shape[0]
				while (u, j) in self.trainDict or j in neglist:
					j = np.random.randint(self.shape[1]) + self.shape[0]
				neglist.add(j)
				tmp_user.append(u)
				tmp_item.append(j)
			user.append(tmp_user)
			item.append(tmp_item)
		return [np.array(user), np.array(item)]

	def writeGraph(self, filename):
		with open(filename, 'w+') as f:
			for (u, v) in self.graph.edges():
				f.write(str(u) + " " + str(v) + " " + str(self.graph[u][v]['weight']) + "\n")
			# print(u, v, self.graph[u][v]['weight'])
