import numpy as np
import torch
import networkx as nx
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

samplling_probability = 0.05

class Dataset(object):
	def __init__(self, user_rating_dict, user_review_dict, item_user_dict):
		self.graph = nx.Graph()
		self.data, self.adj, self.model_D2V = self.getData(
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

		tagged_data = []
		feature = []
		u = 0
		for user, reviews in user_review_dict.items():
			u += 1
			str = ""
			for each in reviews:
				str += " "
				str += each[1]
			tagged_data.append(TaggedDocument(words=word_tokenize(str), tags=[nodedict[user]]))

		i = 0
		for item, reviews in item_user_dict.items():
			i += 1
			str = ""
			for each in reviews:
				str += " "
				str += each[1]
			tagged_data.append(TaggedDocument(words=word_tokenize(str), tags=[nodedict[item]]))

		self.shape = [u, i]
		print("user: {} item: {}".format(u, i))
		self.graph.add_nodes_from(range(u+i))

		for user, rating in user_rating_dict.items():
			for each in rating:
				data.append((nodedict[user], nodedict[each[0]], each[1]))
				if each[1] > maxr:
					maxr = each[1]
				self.graph.add_weighted_edges_from([(nodedict[user], nodedict[each[0]], each[1])])

		self.maxRate = maxr

		max_epochs = 1
		vec_size = 20
		alpha = 0.025

		model_D2V = Doc2Vec(dm=1,
		                alpha=alpha,
		                vector_size=vec_size,
		                min_alpha=0.00025,
	                    min_count=1)

		model_D2V.build_vocab(tagged_data)

		for epoch in range(max_epochs):
			model_D2V.train(tagged_data,
		                total_examples=(u+i),
		                epochs=model_D2V.epochs)
			# decrease the learning rate
			model_D2V.alpha -= 0.0002
			# fix the learning rate, no decay
			model_D2V.min_alpha = model_D2V.alpha
			if (epoch + 1) % 10 == 0:
				print('Doc2Vec iteration {0}'.format(epoch + 1))

		adj_matrix = nx.adjacency_matrix(self.graph)
		e = list(self.graph.edges.data())
		adj = torch.zeros(u + i, u + i)

		for edge in e:
			adj[edge[0]][edge[1]] = edge[2]['weight']
			adj[edge[1]][edge[0]] = edge[2]['weight']
		adj = adj.type(torch.FloatTensor)

		return data, adj, model_D2V

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
				j = np.random.randint(self.shape[1])
				while (i[0], j) in self.trainDict:
					j = np.random.randint(self.shape[1])
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
				j = np.random.randint(self.shape[1])
				while (u, j) in self.trainDict or j in neglist:
					j = np.random.randint(self.shape[1])
				neglist.add(j)
				tmp_user.append(u)
				tmp_item.append(j)
			user.append(tmp_user)
			item.append(tmp_item)
		return [np.array(user), np.array(item)]
