import pickle
import numpy as np 
import datetime
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import sys
sys.path.append('D:/360安全浏览器下载/network_science_lab/presentation 3/second_round')
from data_preprocess import transform_count_into_prob, establish_two_dict, pickle_data, read_pickle # function from my data_process file
from collections import Counter
import os
from gensim.models import Word2Vec

# based on parquet dataset as follows:
# voter1  voter2   # of shared nft
#   ...      ...        ...
# we build a dictionary such that all
# voters are keys and all other voters who share some nft
# are values in such a dict
def preprocess():

	data = pq.read_table(u'D:/360安全浏览器下载/network_science_lab/presentation 4/voter similarity 2/dao_voters_similarity_numeric.pq').to_pandas()

	voter1 = data['voter1']
	voter2 = data['voter2']
	shared_nft = data['nshareddaos']

	temp_dict = {}
	count = 0

	for i, j, z in zip(voter1, voter2, shared_nft):

		if z == 0.0: pass
		else:

			if temp_dict.get(i) is None:

				temp_dict[i] = {j:z}

			else:

				temp_dict.get(i)[j] = z

			if temp_dict.get(j) is None:

				temp_dict[j] = {i:z}

			else:

				temp_dict.get(j)[i] = z

		count += 1
		print(count)
		#if count == 100: break

	

	return temp_dict


# dict is just the output in last function, which is
# dictionary in which keys are voters and values are
# all other voters who share the kinds of nft
# data is just the input to preprocess function above
# which is in form of
# voter1  voter2   # of shared nft
#   ...      ...        ...
# ratio is set to deal with a situation such that
# A - C but meanwhile
# A - B - C
# the logic is that for (A,C), based on dict figured out above
# we get all voters sharing nft with A and C, voter_A, voter_C
# set(voter_A) & set(voter_C), we have intersection of both
# then we can take further operations
# it returns a dataframe in a form of
# voter1 voter2  mutliplication   summation    max   min
#   ...     ...           ...           ...     ..    ..
def main_process(dict_, data, ratio: int = 0.5):

	
	voter1 = data['voter1']
	voter2 = data['voter2']
	shared_nft = data['nshareddaos']
	# print(data)

	count = 0 
	sum_ = []
	max_ = []
	min_ = []
	mult_ = []

	
	for i, j, k in zip(voter1, voter2, shared_nft):

		temp = dict_.get(i).keys() & dict_.get(j).keys()
		len_ = len(temp)
		temp1 = 0
		temp2 = 0

		

			
		if len_ == 0: pass

		else:

			for z in temp:

				temp1 += dict_.get(i).get(z)
				temp2 += dict_.get(j).get(z)

			temp1 = temp1 / len_
			temp2 = temp2 / len_

		if k == 0.0:



			sum_.append(sum([temp1, temp2]))
			max_.append(max(temp1, temp2))
			min_.append(min(temp1, temp2))
			mult_.append(temp1*temp2)

		else:

			if len_ == 0:

				sum_.append(k)
				max_.append(k)
				min_.append(k)
				mult_.append(k)

			else:

				sum_.append(sum([temp1, temp2]) * (1 - ratio) + k * ratio)
				max_.append(max(temp1, temp2) * (1 - ratio) + k * ratio)
				min_.append(min(temp1, temp2) * (1 - ratio) + k * ratio)
				mult_.append(temp1*temp2 * (1 - ratio) + k * ratio)


		count += 1
		print(count)
		#if count == 10: break

	df = pd.DataFrame({
		'voter1':voter1, 'voter2':voter2, 'mutliplication':mult_,
		'summation':sum_, 'maximum':max_, 'minimum':min_
		})


	return df



	



			

		

	




	









