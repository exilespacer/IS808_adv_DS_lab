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
voter_similarity = data_dir / "voter_similarity_by_siaoXu.csv"
similarity_by_category = data_dir / "similarity_by_category.pq"

output_file = "voter_dao_similarity_mapping.pq"

# %% Read Data sets

df_dao_voter_similarity_binary = pd.read_parquet(dao_voter_similarity_binary)
df_dao_voter_similarity_numeric = pd.read_parquet(dao_voter_similarity_numeric)
df_voter_similarity = pd.read_csv(voter_similarity)
df_similarity_by_category = pd.read_parquet(similarity_by_category)

# %% Data set info/statistics
# df_voter_similarity.info()
# print(df_voter_similarity.columns.tolist())

# %%
merged1 = pd.merge(df_dao_voter_similarity_binary,df_dao_voter_similarity_numeric,on=['voter1', 'voter2'])
merged2 = pd.merge(merged1,df_voter_similarity,on=['voter1', 'voter2'])
merged3 = pd.merge(merged2,df_similarity_by_category,on=['voter1', 'voter2']).drop_duplicates(
   subset = ['voter1', 'voter2'],
   keep = 'last').reset_index(drop = True)

# merged = pd.merge(
#     df_dao_voter_similarity_binary,
#     df_dao_voter_similarity_numeric,
#     df_voter_similarity,
#     df_similarity_by_category,

#     left_on="voter1" "voter2",
#  #   right_on="",
#     how="left"
#     ).drop_duplicates("voter1", "voter2")

merged3.to_parquet(
    data_dir / output_file, index=False, compression="brotli"
)