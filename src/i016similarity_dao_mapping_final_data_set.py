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

cosine_similarity = data_dir / "cosine_similarity.pq"
num_of_shared_NFT = data_dir / "num_of_shared_NFT.pq"

similarity_by_category = data_dir / "similarity_by_category.pq"

output_file = "similarity_dao_mapping_final_data_set.pq"

# %% Read Data sets

df_dao_voter_similarity_binary = pd.read_parquet(dao_voter_similarity_binary)
df_dao_voter_similarity_numeric = pd.read_parquet(dao_voter_similarity_numeric)

df_cosine_similarity = pd.read_parquet(cosine_similarity)
df_cosine_similarity.columns = ['voter1', 'voter2', 'cosine_similarity']
df_num_of_shared_NFT = pd.read_parquet(num_of_shared_NFT)
df_num_of_shared_NFT.columns = ['voter1', 'voter2', 'number_of_shared_NFT']

df_similarity_by_category = pd.read_parquet(similarity_by_category)

# %% Data set info/statistics
df_dao_voter_similarity_binary.info()
len(df_dao_voter_similarity_binary.index)

df_dao_voter_similarity_numeric.info()
len(df_dao_voter_similarity_numeric.index)

df_cosine_similarity.info()
len(df_cosine_similarity.index)

df_num_of_shared_NFT.info()
len(df_num_of_shared_NFT.index)

df_similarity_by_category.info()
len(df_similarity_by_category.index)

# %% Merge data set


merged1 = pd.merge(df_dao_voter_similarity_binary,df_dao_voter_similarity_numeric, how='left', left_on=['voter1', 'voter2'], right_on=['voter1', 'voter2']).fillna(0)
merged2 = pd.merge(merged1,df_cosine_similarity, how='left', left_on=['voter1', 'voter2'], right_on=['voter1', 'voter2']).fillna(0)
merged3 = pd.merge(merged2,df_num_of_shared_NFT, how='left', left_on=['voter1', 'voter2'], right_on=['voter1', 'voter2']).fillna(0)
merged_final = pd.merge(merged3,df_similarity_by_category, how='left', left_on=['voter1', 'voter2'], right_on=['voter1', 'voter2']).fillna(0).drop_duplicates(
    subset = ['voter1', 'voter2'],
    keep = 'last').reset_index(drop = True)  

merged_final["similarity_Art"] = merged_final["similarity_Art"].astype(int)
merged_final["similarity_Collectible"] = merged_final["similarity_Collectible"].astype(int)
merged_final["similarity_Games"] = merged_final["similarity_Games"].astype(int)
merged_final["similarity_Other"] = merged_final["similarity_Other"].astype(int)
merged_final["similarity_Utility"] = merged_final["similarity_Utility"].astype(int)

merged_final['similarity_total'] = merged_final['similarity_Art'] + merged_final['similarity_Collectible'] + merged_final['similarity_Games'] + merged_final['similarity_Other'] + merged_final['similarity_Utility']

merged_final.info()   
merged_final.head() 

merged_final.to_parquet(
    data_dir / output_file, index=False, compression="brotli"
 )