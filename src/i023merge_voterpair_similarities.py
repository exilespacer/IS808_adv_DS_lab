# %%
from pathlib import Path
# %% Path
# Duplicate and comment out this row
# It should be the top level folder of the repository
# Sven
# projectfolder = Path("/project/IS808_adv_DS_lab")
# Mengnan
projectfolder = Path("C:\clone\IS808_adv_DS_lab")
# Chia-Yi
# projectfolder = Path("xyz/abc/IS808_adv_DS_lab")

# %%
data_dir = projectfolder / "data"
dir_path = "vis"
# %% library
# %%
import os
import sys

os.chdir(projectfolder)  # For interactive testing
sys.path.insert(0, "")  # Required for loading modules


import io
import json
import pickle
from itertools import permutations

import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import permutations
from tqdm import tqdm

# %% Load Data --Dyad 
df_covoting1 = pd.read_parquet(data_dir/"dao_voters_similarity_votechoice_binary.pq").sort_values(["voter1","voter2"]).set_index(["voter1","voter2"])
df_covoting1.info()
df_covoting1.head()

df_covoting2 = pd.read_parquet(data_dir/"dao_voters_similarity_votechoice_numeric.pq").sort_values(["voter1","voter2"]).set_index(["voter1","voter2"])
df_covoting2.info()
df_covoting2.head()

df_covoting3 = pd.read_parquet(data_dir/"dao_voters_similarity_votechoice_normalized_numeric.pq").sort_values(["voter1","voter2"]).set_index(["voter1","voter2"])
df_covoting3.info()
df_covoting3.head()
df_covoting3.describe()
# %%
df_codao_dummy = pd.read_parquet(data_dir/"dao_voters_similarity_binary.pq").sort_values(["voter1","voter2"]).set_index(["voter1","voter2"])
df_codao_dummy.info()
df_codao_dummy.head()
df_codao_dummy.describe()

df_codao_numeric = pd.read_parquet(data_dir/"dao_voters_similarity_numeric.pq").sort_values(["voter1","voter2"]).set_index(["voter1","voter2"])
df_codao_numeric.info()
df_codao_numeric.head()
df_codao_numeric.describe()

df_codao_n_normalized = pd.read_parquet(data_dir/"dao_voters_similarity_numeric_normalized.pq").sort_values(["voter1","voter2"]).set_index(["voter1","voter2"])
df_codao_n_normalized.info()
df_codao_n_normalized.head()
df_codao_n_normalized.describe()
# %%
""" df_similarity0 = pd.read_parquet(data_dir/"similarity_by_nft_kinds.pq").rename(
    lambda x: x.replace(" ","").lower(),axis = 1
)
df_similarity0 =df_similarity0.sort_values(["voter1","voter2"]).set_index(["voter1","voter2"])
df_similarity0.info()
df_similarity0
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df_similarity0) """

df_similarity1 = pd.read_parquet(data_dir/"similarity_by_category.pq").sort_values(["voter1","voter2"]).set_index(["voter1","voter2"])
df_similarity1.info()
df_similarity1.head()

df_similarity2 = pd.read_parquet(data_dir/"similarity_distance_by_category.pq").sort_values(["voter1","voter2"]).set_index(["voter1","voter2"])
df_similarity2.info()
df_similarity2.head()

df_similarity3 = pd.read_parquet(data_dir/"similarity_by_nft_kinds.pq").sort_values(["voter1","voter2"]).set_index(["voter1","voter2"])
df_similarity3.info()
df_similarity3.head()

df_similarity4 = pd.read_parquet(data_dir/"similarity_distance_by_nft.pq").sort_values(["voter1","voter2"]).set_index(["voter1","voter2"])
df_similarity4.info()
df_similarity4.head()


# %%
df_1st_similarity = pd.read_parquet(data_dir/"similarity_by_nft_kinds_1st_degree.pq").sort_values(["voter1","voter2"]).set_index(["voter1","voter2"])
df_1st_similarity.info()
df_1st_similarity.head()

df_2nd_similarity1 = pd.read_parquet(data_dir/"similarity_by_nft_kinds_2nd_degree_average.pq").sort_values(["voter1","voter2"]).set_index(["voter1","voter2"])
df_2nd_similarity1.info()
df_2nd_similarity1.head()

df_2nd_similarity2 = pd.read_parquet(data_dir/"similarity_by_nft_kinds_2nd_degree_average.pq").sort_values(["voter1","voter2"]).set_index(["voter1","voter2"])
df_2nd_similarity2.info()
df_2nd_similarity2.head()

# %% Load Data --Node
def get_df_raw():
    df_opensea =pd.read_parquet(data_dir/"opensea_collections.pq")
    df_opensea.info()
    df_opensea.head()

    df_dao_voters = pd.read_parquet(data_dir/"dao_voter_mapping.pq")
    df_dao_voters.info()
    df_dao_voters.head()

    df_opensea_categories_top50 = pd.read_parquet(data_dir/"opensea_categories_top50.pq")
    ids = df_opensea_categories_top50 ['slug']
    df_opensea_categories_top50[ids.isin(ids[ids.duplicated()])].sort_values("slug")

    df_opensea_categories_top50 = df_opensea_categories_top50.drop_duplicates(subset=['slug'])
    df_opensea_categories_top50.info()
    df_opensea_categories_top50.head()

    df_filter = pd.read_parquet(data_dir/"relevant_voters_with_voterid.pq") # 1124 voterid
    df_filter.info()
    df_filter.head()

    df_voter_slug_category = pd.read_parquet(data_dir/"voter_slug_category.pq")
    df_slug_category = df_voter_slug_category.drop_duplicates(subset=['slug']).drop("voterid", axis=1)
    df_slug_category.info()
    df_slug_category.head()


    df_raw1 = pd.merge(
        df_dao_voters,
        df_opensea,
        on="voterid",
        how="inner",
        validate="m:m",
    )

    df_raw2 = pd.merge(
        df_raw1,
        df_slug_category,
        on="slug",
        how="left",
        validate="m:1",
    )

    df_raw = pd.merge(
        df_raw2,
        df_filter,
        on="voterid",
        how="inner",
        validate="m:1",
    )
    df_raw.info()
    df_raw.head()
    return df_raw

df_raw = get_df_raw() 
df_raw.to_csv(f"{data_dir}/df_raw.csv", index=False)

df_voter_nft = df_raw[['voterid', 'slug']]
df_voter_nft.to_csv(f"{data_dir}/df_voter_nft.csv", index=False)

# # Voter network
df_voter_nft =pd.read_csv(f"{data_dir}/df_voter_nft.csv")

df = pd.read_parquet(data_dir/"regression_frame_merged.pq").sort_values(["voter1","voter2"]).set_index(["voter1","voter2"])
df.head()
# df_network_voter = df_network_voter.dropna

df_network_voter_full = df.merge(
    df_codao_dummy, how = 'left', on =['voter1','voter2']).merge(
    df_codao_numeric, how = 'left', on =['voter1','voter2']).merge(
    df_codao_n_normalized, how = 'left', on =['voter1','voter2']).merge(    
    df_covoting1,how = 'left', on =['voter1','voter2']).merge(
    df_covoting2,how = 'left', on =['voter1','voter2']).merge(
    df_covoting3, how = 'left', on =['voter1','voter2']).merge(
    df_similarity1,how = 'left', on =['voter1','voter2']).merge(
    df_similarity2,how = 'left', on =['voter1','voter2']).merge(
    df_similarity3,how = 'left', on =['voter1','voter2']).merge(
    df_similarity4,how = 'left', on =['voter1','voter2']
    ) 



""" df_network_voter_full = df_network_voter.merge(
    df_codao_dummy,how = 'left', on =['voter1','voter2']).merge(
    df_covoting1,how = 'left', on =['voter1','voter2']).merge(
    df_covoting0,how = 'left', on =['voter1','voter2']).merge(
    df_similarity0,how = 'left', on =['voter1','voter2']).merge(
    df_similarity2,how = 'left', on =['voter1','voter2']
    ) """

df_network_voter_full = df_network_voter_full*1
# df_network_voter_full = df_network_voter_full.fillna(0)
df_network_voter_full.to_csv(f"{dir_path}/vis_network_voter_edges.csv", index=True)


""" # %% voter- distinct nft
network_voters = {}
for grp, df_grp in tqdm(df_voter_nft.groupby("slug")):
    for p in permutations(sorted(df_grp.voterid.unique()), 2):
        if p not in network_voters:
            network_voters[p] = 0
        network_voters[p] += 1

# %%
df_network_voter = pd.DataFrame(
    [
        {"voter1": source, "voter2": target, "n_slug": weight}
        for (source, target), weight in network_voters.items()
    ]
)
df_network_voter = df_network_voter.sort_values(['voter1','voter2']).set_index(['voter1','voter2']) """