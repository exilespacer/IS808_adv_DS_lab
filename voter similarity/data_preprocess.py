import pickle
import numpy as np 
import datetime


def read_pickle(path):

	f2 = open(str(path), 'rb')
	output = pickle.load(f2)
	f2.close()

	return output

def pickle_data(input, path):

	f = open(str(path),'wb')  
	pickle.dump(input,f)  
	f.close() 

	return True

# I establish two dictorinary-type datasets
# one is voters(key) -> NFT(values)
# another NFT(keys) -> voters(values)
# the benefit of doing so is to facilitate to simulate
# random walk on a bipartite network
# input：  
# voters: list of voters   
# slugs: list of NFT
# quantity: the number of NFT that a specific voter once bought 
# renturn two dictionaries (1) voters -> NFT and quantity that he/she holds
# (2) NFT -> voters and quantity that he/she holds
def establish_two_dict(voters, slugs, quantity):

	address = voters 
	slug = slugs 
	quantity = quantity 

	# del data

	count = 0
	address_to_products = {}
	products_to_address = {}
	

	for i, j, k in zip(address, slug, quantity):

		if address_to_products.get(i) is None:

			address_to_products[i] = [[j], [int(k)]]

		else:

			temp = address_to_products.get(i)
			temp_1 = temp[0]
			temp_2 = temp[1]

			if j not in temp_1: 

				temp_1.append(j)
				temp_2.append(int(k))

			else:

				temp_2[temp_1.index(j)] += int(k)


			address_to_products[i] = [temp_1, temp_2]

		if products_to_address.get(j) is None:

			products_to_address[j] = [[i], [int(k)]]

		else:

			temp = products_to_address.get(j)
			temp_1 = temp[0]
			temp_2 = temp[1]

			if i not in temp_1: 

				temp_1.append(i)
				temp_2.append(int(k))

			else:

				temp_2[temp_1.index(i)] += int(k)


			products_to_address[j] = [temp_1, temp_2]


		print(count)
		count+=1

		# if count == 1000: break


	return address_to_products, products_to_address





# since we focus on top 20 NFT only right now, 
# from the dictionary having all NFT, we wish to
# sift out top 20 NFTs
# 2 inputs, one is dictionary of NFT -> voters
# another list of top 20 NFT
def sift_out_top20NFT(original_dataset, top_list):

	temp_dict = {}
	count = 0
	for i in top_list:

		print(count)
		count += 1
		temp_dict[i] = original_dataset.get(i) 

	return temp_dict



# for two dictionaries, we have value data in form of [[object], [count]]
# we transform it into [[object], [prob]]
# prob = count / sum(count)
def transform_count_into_prob(data):

	count = 0

	for i in data:

		#print(i)
		temp = data.get(i)[1]

		data.get(i)[1] = np.array(temp) / sum(np.array(temp))
		
		count += 1
		print(count)
		#if count == 10: break
		#print(data.get(i))

	return data