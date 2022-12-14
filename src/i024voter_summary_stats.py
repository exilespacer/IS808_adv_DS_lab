# %%
from pathlib import Path

# Duplicate and comment out this row
# It should be the top level folder of the repository
# Sven
# projectfolder = Path("/project/IS808_adv_DS_lab")
# Chia-Yi
# projectfolder = Path("xyz/abc/IS808_adv_DS_lab")
# Mengnan 

projectfolder = Path("C:\clone\IS808_adv_DS_lab")

# %%
# ! pip install pyarrow

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
from pyvis.network import Network
from tqdm import tqdm


# %%
data_dir = projectfolder / "data"
dir_path = projectfolder / "vis"
# !pip install tqdm
# !pip install pyvis


# %% [markdown]
# # Voter network
df_raw =pd.read_csv(f"{data_dir}/df_raw.csv")

# %% [markdown]
# # Summary statistics of data

# %% [markdown]
# ## distinct N
df_N = (
    df_raw.loc[:, ["dao", "voterid", "slug","category"]]
    .nunique()
    .to_frame("distinct N")
    .T
)
df_N.columns = ["DAO", "voter", "NFT","NFT category"]
df_N.head()
df_N.pipe(display)


df_voter = df_raw.groupby("voterid").agg(
    N_nft_kinds=("slug", "nunique"),
    N_nft_categoty=("category", "nunique"),
    N_nft_quantity=("owned_asset_count", np.sum),
)
df_voter.head()

df_voter_nodes =pd.read_csv(f"{data_dir}/voter_taste_category.csv")
df_voter_nodes.count()
df_voter_nodes.head()

df_network_voter_nodes = pd.merge(
    df_voter,
    df_voter_nodes,
    left_on="voterid",
    right_on="voter",
    how="inner",
    validate="1:1",
)
df_network_voter_nodes.rename({'voter': 'voterid'}, axis=1, inplace=True)
df_network_voter_nodes = df_network_voter_nodes.set_index(['voterid'])
df_network_voter_nodes.count()
df_network_voter_nodes.head()
df_network_voter_nodes.to_csv(f"{dir_path}/vis_network_voter_nodes.csv", index=True)


df_voter_stats = (
    df_voter.assign(
        N_nft_quantity_wei=lambda x: x.N_nft_quantity.div(10**18),
        log_N_nft_quantity=lambda x: x.N_nft_quantity.map(np.log),
    )
    .describe()
    .assign(
        N_nft_quantity=lambda x: x.N_nft_quantity.map(lambda x: f"{x:.2e}"),
        N_nft_quantity_wei=lambda x: x.N_nft_quantity_wei.map(lambda x: f"{x:.2e}"),
        N_nft_kinds=lambda x: x.N_nft_kinds.map(lambda x: f"{x:.0f}"),
        N_nft_categories=lambda x: x.N_nft_categoty.map(lambda x: f"{x:.0f}"),
        log_N_nft_quantity=lambda x: x.log_N_nft_quantity.map(lambda x: f"{x:.2f}"),
        
    )
    .loc[
        ["mean", "std", "min", "25%", "50%", "75%", "max"],
        ["N_nft_kinds",  "N_nft_categories", "log_N_nft_quantity", "N_nft_quantity"],
    ]
)
df_voter_stats.columns = [
    "N(distinct NFT)",
    "N(distinct NFT categories)",
    "log(Total NFT collections)",
    "Total NFT collections",
   
]
df_voter_stats.pipe(display)