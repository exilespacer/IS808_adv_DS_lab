import pickle
import numpy as np 
import datetime
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import sys
# the path points to the document of "data_preprocess.py"
sys.path.append('D:/360安全浏览器下载/network_science_lab/presentation 3/second_round')
from data_preprocess import transform_count_into_prob, establish_two_dict, pickle_data, read_pickle # function from my data_process file
from collections import Counter
import os
from gensim.models import Word2Vec


# produce two dict_type datasets
# data = pq.read_table(u'D:/360安全浏览器下载/network_science_lab/presentation 4/opensea_collections.pq').to_pandas()
# address = data['voterid']
# nft = data['slug']
# quantity = data['owned_asset_count']
def step_1(address, nft, quantity):

	address_to_products, products_to_address = establish_two_dict(address, nft, quantity)

	return address_to_products, products_to_address


# transform two dicts with counts into dicts with probability
# input--"address_slug, slug_address" are two outputs from function step1
def step_2(address_slug, slug_address):

	address_slug = transform_count_into_prob(address_slug)
	slug_address = transform_count_into_prob(slug_address)

	return address_slug, slug_address


# first-round sampling, draw samples from voters to NFT
# "address_slug" is one of two output from function step2
def step_3(address_slug):

	first_dict = {}
	count = 1
	temp_set = set(address_slug.keys())

	for i in temp_set:		

		temp = list(address_slug.get(i))

		#print(i)

		p = np.array(temp[1], dtype='float64')
		#print(p)

		first_dict[i] = Counter(np.random.choice(temp[0], size = 1000, p = p))
		del address_slug[i]
	
		#print(type(first_dict.get(i)))

		count += 1 
		print(count)
		
		# because the dataset is so huge that overload my internal memory
		# i have got to save the outputs separately so as to free some memory
		# during running the program
		# if you have sufficient memory, you can combine several steps
		# into one and output one result
		if count % 50000 == 0:

			pickle_data(first_dict, 'D:/360安全浏览器下载/network_science_lab/presentation 4/sampling results/result_from_first_sampling' + str(count))
			first_dict = {}

	pickle_data(first_dict, 'D:/360安全浏览器下载/network_science_lab/presentation 4/sampling results/result_from_first_sampling' + str(count))

	return True

# from step3, I save a piece of output sequentially for avoid running out of memory
# this function is to combine all partial dataset from step3 into one
# this step could be ignored if you don't want to combine all data into one
# my function step_5_auxiliary just read in dataset one by one and deal with
# them one by one also
# if you have sufficient memory, you don't need to do so
def step_4():

	# read in saved data from step3
	d1 = read_pickle('D:/360安全浏览器下载/network_science_lab/presentation 4/sampling results/result_from_first_sampling50000')
	d2 = read_pickle('D:/360安全浏览器下载/network_science_lab/presentation 4/sampling results/result_from_first_sampling100000')
	d3 = read_pickle('D:/360安全浏览器下载/network_science_lab/presentation 4/sampling results/result_from_first_sampling150000')
	d4 = read_pickle('D:/360安全浏览器下载/network_science_lab/presentation 4/sampling results/result_from_first_sampling200000')
	d5 = read_pickle('D:/360安全浏览器下载/network_science_lab/presentation 4/sampling results/result_from_first_sampling250000')
	d6 = read_pickle('D:/360安全浏览器下载/network_science_lab/presentation 4/sampling results/result_from_first_sampling300000')
	d7 = read_pickle('D:/360安全浏览器下载/network_science_lab/presentation 4/sampling results/result_from_first_sampling350000')
	d8 = read_pickle('D:/360安全浏览器下载/network_science_lab/presentation 4/sampling results/result_from_first_sampling400000')
	d9 = read_pickle('D:/360安全浏览器下载/network_science_lab/presentation 4/sampling results/result_from_first_sampling450000')
	d10 = read_pickle('D:/360安全浏览器下载/network_science_lab/presentation 4/sampling results/result_from_first_sampling453886')

	data = {**d1, **d2, **d3, **d4, **d5, **d6, **d7, **d8, **d9, **d10}
	pickle_data(data, 'D:/360安全浏览器下载/network_science_lab/presentation 4/sampling results/result_from_first_sampling_all')

	return True


# read in all voter counts of samples
# function step5 is a part of this function
def step_5_auxiliary():

	slug_address = read_pickle('D:/360安全浏览器下载/network_science_lab/presentation 4/two dict dataset/slug_to_voter_prob')
	aaa = {
	'result_from_first_sampling50000', 'result_from_first_sampling100000',
	'result_from_first_sampling150000', 'result_from_first_sampling200000',
	'result_from_first_sampling250000', 'result_from_first_sampling300000',
	'result_from_first_sampling350000', 'result_from_first_sampling400000',
	'result_from_first_sampling450000', 'result_from_first_sampling453886'
	}
	count = 0
	for i in aaa:

		# function defined below
		step_5(i, slug_address, count)
		count+=1

	return True

# second round sampling and count up
# first_round_count is output derived from last step 
# slug_address is one of two output from function step2
# draw samples from NFT back to voters
def step_5(first_round_count, slug_address, num):


	ccc = {}

	count = 1
	dd = set(first_round_count.keys())
	for i in dd:

		# save the data partly in order to avoid running out of memory
		if count % 20000 == 0: 

			pickle_data(ccc, 'D:/360安全浏览器下载/network_science_lab/presentation 4/second sampling/second_round_count_step' + str(num) + '_' + str(count))
			ccc = {}

		temp1 = first_round_count.get(i)
		temp2 = []

		for z in temp1.keys():

			index = temp1.get(z)
			#print(z)
			#print(index)
			#print(type(index))

			temp = slug_address.get(z)

			p = np.array(temp[1], dtype='float64')				
				
			try: temp2 += list(np.random.choice(temp[0], size = index, p = p))
			except ValueError:
				p = np.array(temp[1], dtype='float64') * (-1)
				p = p / sum(p)
				temp2 += list(np.random.choice(temp[0], size = index, p = p))
			else: pass

		ccc[i] = Counter(temp2)
		del first_round_count[i]
		count+=1
		print(count)


	del temp
	del temp2
	del temp1
	del slug_address
	del first_round_count
	pickle_data(ccc, 'D:/360安全浏览器下载/network_science_lab/presentation 4/second sampling/second_round_count_step' + str(num) + '_' + str(count)) 

	return True


# the function defined as a part of function step_6 below
# this and step_6 both form the edges of each pair of nodes
# data is just counts of second sampling derived from step_5_auxiliary
def form_pairs(path, nodes_in_new_network, threshold: int=50):

	data = read_pickle(path)

	count = 0

	#nodes_in_new_network = set()

	for i in data:

		temp = data.get(i)
		for j in temp:

			if temp.get(j) > threshold and (j, i) not in nodes_in_new_network and i != j:

				nodes_in_new_network.add((i, j))
		count += 1
		print(count)

		#if count == 5: break

	return nodes_in_new_network

def step_6(threshold: int = 10):
	

	set_ = set()
	path = 'D:/360安全浏览器下载/network_science_lab/presentation 4/second sampling/'

	for files in os.listdir(path):

		set_ = form_pairs(path + files, set_, threshold=threshold)

	
	#set_ = set()
	#path = 'D:/360安全浏览器下载/network_science_lab/presentation 4/second sampling/second_round_count_step0_20000'
	#set_ = form_pairs(path, set_, threshold=threshold)

	count = 0
	_set = set()
	if len(set_) % 2 == 0:

		temp_count = len(set_) / 2

	else: temp_count = (len(set_) - 1) / 2

	temp_set = list(set_)
	for i in temp_set:

		_set.add(i)
		set_.remove(i)
		count += 1
		print(count)

		if count == temp_count: break

	del temp_set

	# save output separately
	pickle_data(_set, 'D:/360安全浏览器下载/network_science_lab/presentation 4/new_network/graph_pairs1')
	del _set
	pickle_data(set_, 'D:/360安全浏览器下载/network_science_lab/presentation 4/new_network/graph_pairs2')

	return True

# function for computing cosine similarity
def cosine_similarity(x,y):

    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0.0: return 'nan'

    return num/denom



# data is just something from step_6 which is a set of nodes edges
def train_node2vec(pairs):

	# construct framework of net
	G = nx.Graph()
	G.add_edges_from(pairs)
	del pairs

	node2vec = Node2Vec(G, dimensions=5, walk_length=3, num_walks=2, workers=1)
	model = node2vec.fit(window=10, min_count=1, batch_words=4)
	# save the trained model and embedding nodes
	model.save('drive/My Drive/network science/nodes embedding')
	model.wv.save_word2vec_format('drive/My Drive/network science/nodes embedding2')

	return model


# input is just output from last function "train_node2vec"
# graph_pairs is output from step_6, which is nodes pairs
# voter_slug is one of two dict we produced at very beginning
# voter_slug is to compute # of common NFT two voters have together
# the output of this function is as follows:

#   voter 1    voter 2    cosin_similarity      # of sorts of shared NFT
#    ...         ...         ...                     ...
#    ...         ...         ...                     ...
# data is saved in form of parquet
def step_7(model, graph_pairs, voter_slug):


	data = graph_pairs
	
	voter_1 = []
	voter_2 = []
	similarity = []
	count = 0

	for i in data:

		voter_1.append(i[0])
		voter_2.append(i[1])

		temp1 = model.wv[str(i[0])]
		temp2 = model.wv[str(i[1])]
		similarity.append(cosine_similarity(temp1, temp2))
		print(count)

		count += 1

	temp_a = []
	temp_b = []
	temp_c = []
	temp_d = []
	count = 0
	temp_e = []
	temp_f = []

	for i, j, z in zip(voter_1, voter_2, similarity):

		if i == j: pass
		else:

			temp_a.append(i)
			temp_b.append(j)
			temp_c.append(z)
			temp_1 = set(voter_slug.get(i)[0])
			temp_2 = set(voter_slug.get(j)[0])
			temp_d.append(len(temp_1 & temp_2))

		count += 1
		print(count)


	df = pd.DataFrame({
		'voter 1':temp_a, 'voter 2':temp_b, 'cosine similarity':temp_c,
		'number of shared NFT':temp_d
		})


	df.to_parquet(
        'D:/360安全浏览器下载/network_science_lab/presentation 4/similarity_and_sharedNFT_based_NFTcollection.pq',
         compression="brotli", index=False
    )


	return df








	


