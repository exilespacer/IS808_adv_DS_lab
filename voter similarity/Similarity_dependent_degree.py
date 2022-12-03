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

	#print(temp_dict)
	pickle_data(temp_dict, 'D:/360安全浏览器下载/network_science_lab/presentation 4/voter similarity 2/dict_shared_nft')

	return True

def main_process(ratio: int = 0.5):

	dict_ = read_pickle('D:/360安全浏览器下载/network_science_lab/presentation 4/voter similarity 2/dict_shared_nft')
	data = pq.read_table(u'D:/360安全浏览器下载/network_science_lab/presentation 4/voter similarity 2/dao_voters_similarity_numeric.pq').to_pandas()
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

			temp1 = round(temp1 / len_, 3)
			temp2 = round(temp2 / len_, 3)

		if k == 0.0:



			sum_.append(sum([temp1, temp2]))
			max_.append(max(temp1, temp2))
			min_.append(min(temp1, temp2))
			mult_.append(round(temp1*temp2, 3))

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
				mult_.append(round(temp1*temp2, 3) * (1 - ratio) + k * ratio)


		count += 1
		print(count)
		#if count == 10: break

	df = pd.DataFrame({
		'voter1':voter1, 'voter2':voter2, 'mutliplication':mult_,
		'summation':sum_, 'maximum':max_, 'minimum':min_
		})

	print(df)
	#print(ratio)

	df.to_parquet(
        'D:/360安全浏览器下载/network_science_lab/presentation 4/voter similarity 2/similarity_secondMethod.pq',
         compression="brotli", index=False
    )

    
 

	return df


if __name__ == '__main__':

	#data = main_process(0.1)
	data = pq.read_table(u'D:/360安全浏览器下载/network_science_lab/presentation 4/voter similarity 2/similarity_secondMethod.pq').to_pandas()
	#plt.hist(data['mutliplication'])
	#plt.hist(data['summation'])
	print(data)
	#plt.show()
	data = data['summation']

	fig = plt.figure(figsize=(15, 9))
	data.hist(bins=200)

	plt.xlabel('similarity', fontsize=13)
	plt.ylabel('Frequency')
	plt.tight_layout()
	plt.show()
	



			

		

	




	









