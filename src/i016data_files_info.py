# %%
from pathlib import Path

# Duplicate and comment out this row
# It should be the top level folder of the repository
# Nourhan
projectfolder = Path(r"C:\Users\nshafiks\OneDrive - uni-mannheim.de\Documents\GitHub\IS808_adv_DS_lab")

# %%
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq

os.chdir(projectfolder)  # For interactive testing
sys.path.insert(0, "")  # Required for loading modules

import json

import pandas as pd
from tqdm import tqdm

# %% Final Data sets
data_dir = projectfolder / "data"

dao_voters_similarity_binary = data_dir / "dao_voters_similarity_binary.pq"
dao_voters_similarity_numeric = data_dir / "dao_voters_similarity_numeric.pq"

dao_voters_similarity_votechoice_binary = data_dir / "dao_voters_similarity_votechoice_binary.pq"
dao_voters_similarity_votechoice_numeric = data_dir / "dao_voters_similarity_votechoice_numeric.pq"

similarity_by_nft_kinds = data_dir / "similarity_by_nft_kinds.pq"
similarity_distance_by_nft = data_dir / "similarity_distance_by_nft.pq"

similarity_by_category = data_dir / "similarity_by_category.pq"
similarity_distance_by_category = data_dir / "similarity_distance_by_category.pq"

similarity_secondMethod = data_dir / "similarity_secondMethod.pq"


# %% Read Data sets
df_dao_voters_similarity_binary = pd.read_parquet(dao_voters_similarity_binary)
#df_dao_voter_similarity_binary.columns = ['voter1', 'voter2', 'share_daos']
df_dao_voters_similarity_numeric = pd.read_parquet(dao_voters_similarity_numeric)
#df_dao_voter_similarity_numeric.columns = ['voter1', 'voter2', 'number_shared_daos']

df_dao_voters_similarity_votechoice_binary = pd.read_parquet(dao_voters_similarity_votechoice_binary)
#df_dao_voter_similarity_binary.columns = ['voter1', 'voter2', 'share_daos']
df_dao_voters_similarity_votechoice_numeric = pd.read_parquet(dao_voters_similarity_votechoice_numeric)
#df_dao_voter_similarity_numeric.columns = ['voter1', 'voter2', 'number_shared_daos']

df_similarity_by_nft_kinds = pd.read_parquet(similarity_by_nft_kinds)
#df_similarity_and_sharedNFT_based_NFTcollection.columns = ['voter1', 'voter2', 'cosine_similarity', 'number_of_shared_NFT']
df_similarity_distance_by_nft = pd.read_parquet(similarity_distance_by_nft)
#df_similarity_distance_by_category.columns = ['voter1', 'voter2', 'similarity_category_distance']

df_similarity_by_category = pd.read_parquet(similarity_by_category)
#df_similarity_and_sharedNFT_based_NFTcollection.columns = ['voter1', 'voter2', 'cosine_similarity', 'number_of_shared_NFT']
df_similarity_distance_by_category = pd.read_parquet(similarity_distance_by_category)
#df_similarity_distance_by_category.columns = ['voter1', 'voter2', 'similarity_category_distance']

df_similarity_secondMethod = pd.read_parquet(similarity_secondMethod)
#df_similarity_distance_by_category.columns = ['voter1', 'voter2', 'similarity_category_distance']

# %% Add total category similarity
df_similarity_by_category["similarity_art"] = df_similarity_by_category["similarity_art"].astype(int)
df_similarity_by_category["similarity_collectibles"] = df_similarity_by_category["similarity_collectibles"].astype(int)
df_similarity_by_category["similarity_domain-names"] = df_similarity_by_category["similarity_domain-names"].astype(int)
df_similarity_by_category["similarity_music"] = df_similarity_by_category["similarity_music"].astype(int)
df_similarity_by_category["similarity_photography-category"] = df_similarity_by_category["similarity_photography-category"].astype(int)
df_similarity_by_category["similarity_sports"] = df_similarity_by_category["similarity_sports"].astype(int)
df_similarity_by_category["similarity_trading-cards"] = df_similarity_by_category["similarity_trading-cards"].astype(int)
df_similarity_by_category["similarity_utility"] = df_similarity_by_category["similarity_utility"].astype(int)
df_similarity_by_category["similarity_virtual-worlds"] = df_similarity_by_category["similarity_virtual-worlds"].astype(int)
df_similarity_by_category['similarity_total'] = df_similarity_by_category['similarity_art'] + df_similarity_by_category['similarity_collectibles'] + df_similarity_by_category['similarity_domain-names'] + df_similarity_by_category['similarity_music'] + df_similarity_by_category['similarity_photography-category'] + df_similarity_by_category['similarity_sports']+ df_similarity_by_category['similarity_trading-cards']+ df_similarity_by_category['similarity_utility']+ df_similarity_by_category['similarity_virtual-worlds']
# %% Data set info/statistics
df_dao_voters_similarity_binary.info()
len(df_dao_voters_similarity_binary.index)

df_dao_voters_similarity_numeric.info()
len(df_dao_voters_similarity_numeric.index)

df_dao_voters_similarity_votechoice_binary.info()
len(df_dao_voters_similarity_votechoice_binary.index)

df_dao_voters_similarity_votechoice_numeric.info()
len(df_dao_voters_similarity_votechoice_numeric.index)

df_similarity_by_nft_kinds.info()
len(df_similarity_by_nft_kinds.index)

df_similarity_distance_by_nft.info()
len(df_similarity_distance_by_nft.index)

df_similarity_by_category.info()
len(df_similarity_by_category.index)

df_similarity_distance_by_category.info()
len(df_similarity_distance_by_category.index)

df_similarity_secondMethod.info()
len(df_similarity_secondMethod.index)