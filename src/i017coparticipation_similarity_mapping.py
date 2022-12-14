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

dao_voter_similarity_binary = data_dir / "dao_voters_similarity_binary.pq"
dao_voter_similarity_numeric = data_dir / "dao_voters_similarity_numeric.pq"

similarity_by_nft_kinds = data_dir / "similarity_by_nft_kinds.pq"
similarity_distance_by_nft = data_dir / "similarity_distance_by_nft.pq"

similarity_by_nft_kinds_1st_degree = data_dir / "similarity_by_nft_kinds_1st_degree.pq"
similarity_by_nft_kinds_2nd_degree = data_dir / "similarity_by_nft_kinds_2nd_degree.pq"


similarity_by_category = data_dir / "similarity_by_category.pq"
similarity_distance_by_category = data_dir / "similarity_distance_by_category.pq"

similarity_secondMethod = data_dir / "similarity_secondMethod.pq"

output_file = "coparticipation_similarity_mapping.pq"

# %% Read Data sets
df_dao_voter_similarity_binary = pd.read_parquet(dao_voter_similarity_binary)
df_dao_voter_similarity_binary.columns = ['voter1', 'voter2', 'share_daos']
df_dao_voter_similarity_numeric = pd.read_parquet(dao_voter_similarity_numeric)
df_dao_voter_similarity_numeric.columns = ['voter1', 'voter2', 'number_shared_daos']

df_similarity_by_nft_kinds = pd.read_parquet(similarity_by_nft_kinds)
df_similarity_distance_by_nft = pd.read_parquet(similarity_distance_by_nft)

df_similarity_by_nft_kinds_1st_degree = pd.read_parquet(similarity_by_nft_kinds_1st_degree)
df_similarity_by_nft_kinds_2nd_degree = pd.read_parquet(similarity_by_nft_kinds_2nd_degree)
df_similarity_by_nft_kinds_1st_degree.info()
len(df_similarity_by_nft_kinds_1st_degree.index)
df_similarity_by_nft_kinds_2nd_degree.info()
len(df_similarity_by_nft_kinds_2nd_degree.index)

df_similarity_by_category = pd.read_parquet(similarity_by_category)
df_similarity_distance_by_category = pd.read_parquet(similarity_distance_by_category)

df_similarity_secondMethod = pd.read_parquet(similarity_secondMethod)

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

# %% Merge data set
merged1 = pd.merge(df_dao_voter_similarity_binary,df_dao_voter_similarity_numeric, how='left', left_on=['voter1', 'voter2'], right_on=['voter1', 'voter2']).fillna(0)
merged2 = pd.merge(merged1,df_similarity_by_nft_kinds, how='left', left_on=['voter1', 'voter2'], right_on=['voter1', 'voter2']).fillna(0)
merged3 = pd.merge(merged2,df_similarity_distance_by_nft, how='left', left_on=['voter1', 'voter2'], right_on=['voter1', 'voter2']).fillna(0)
merged4 = pd.merge(merged3,df_similarity_by_category, how='left', left_on=['voter1', 'voter2'], right_on=['voter1', 'voter2']).fillna(0)
merged5 = pd.merge(merged4,df_similarity_distance_by_category, how='left', left_on=['voter1', 'voter2'], right_on=['voter1', 'voter2']).fillna(0)
merged6 = pd.merge(merged5,df_similarity_by_nft_kinds_1st_degree, how='left', left_on=['voter1', 'voter2'], right_on=['voter1', 'voter2']).fillna(0)

merged_final = pd.merge(merged6, df_similarity_by_nft_kinds_2nd_degree, how='left', left_on=['voter1', 'voter2'], right_on=['voter1', 'voter2']).fillna(0).drop_duplicates(
     subset = ['voter1', 'voter2'],
     keep = 'last').reset_index(drop = True)  

merged_final.info()   
merged_final.head() 

merged_final.to_parquet(
     data_dir / output_file, index=False, compression="brotli"
  )