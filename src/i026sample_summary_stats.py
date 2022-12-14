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

# %% Load Data --Node
def get_df_raw():
    df_opensea =pd.read_parquet(data_dir/"opensea_collections.pq")
    df_opensea.info()
    df_opensea.head()

    df_dao_voters = pd.read_parquet(data_dir/"dao_voter_mapping.pq")
    df_dao_voters.info()
    df_dao_voters.head()

    df_raw = pd.merge(
        df_dao_voters,
        df_opensea,
        on="voterid",
        how="inner",
        validate="m:m",
    )

    df_raw.info()
    df_raw.head()
    return df_raw

df_raw = get_df_raw() 
df_raw.head()

# ## distinct N
df_N = (
    df_raw.loc[:, ["dao", "voterid", "slug",'proposalid']]
    .nunique()
    .to_frame("distinct N")
    .T
)
df_N.columns = ["DAO", "voter", "NFT", "proposal"]
df_N.head()
df_N.pipe(display)


df_voter = df_raw.groupby("voterid").agg(
    N_nft_kinds=("slug", "nunique"),
    N_nft_categoty=("category", "nunique"),
    N_nft_quantity=("owned_asset_count", np.sum),
)
df_voter.head()