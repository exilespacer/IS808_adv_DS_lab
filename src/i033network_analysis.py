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
data_dir = projectfolder / "data"
# %%
import os
import sys

os.chdir(projectfolder)  # For interactive testing
sys.path.insert(0, "")  # Required for loading modules


import io
import json
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network

import matplotlib.pyplot as plt


# %% [markdown]
# # Data for network visualization
# - Source
# - Target
# - weight


# %%
dir_path = projectfolder / "vis"
# !pip install tqdm
# !pip install pyvis


# %% [markdown]
# # Voter network
df_network_voter =pd.read_csv(f"{dir_path}/vis_network_voter.csv")
df_network_voter.info()
df_network_voter.count()
df_network_voter.head()

# # Discretion
df_network_voter.mutliplication.describe()
df_network_voter.mutliplication = pd.cut(df_network_voter.mutliplication, bins=6, labels=np.arange(6), right=False)

df_network_voter.similarity_nft_distance.describe()
df_network_voter.similarity_nft_distance = pd.cut(df_network_voter.similarity_nft_distance, bins=6, labels=np.arange(6), right=False)

df_network_voter.similarity_category_distance.describe()
df_network_voter.similarity_category_distance = pd.cut(df_network_voter.similarity_category_distance, bins=6, labels=np.arange(6), right=False)
# %% full
G_voter = nx.from_pandas_edgelist(
    df_network_voter, source="voter1", target="voter2", edge_attr= True,
)

nx.write_gexf(G_voter,f"{dir_path}/gragh_voter.gexf")
with io.open(f"{dir_path}/gragh_voter.nx", mode="wb") as f:
    pickle.dump(G_voter, f)

