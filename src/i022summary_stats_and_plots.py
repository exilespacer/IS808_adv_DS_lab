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
! pip install pyarrow
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

df_opensea =pd.read_parquet(data_dir/"opensea_collections.pq")
df_opensea.info()
df_opensea.head()

df_dao_voters = pd.read_parquet(data_dir/"dao_voter_mapping.pq")
df_dao_voters.info()
df_dao_voters.head()

#df_dao_voters_merged_with_opensea = pd.read_parquet(data_dir/"dao_voters_merged_with_opensea.pq")
#df_dao_voters_merged_with_opensea.info()
#df_dao_voters_merged_with_opensea.head()

df_covoting = pd.read_parquet(data_dir/"covoting_between_voters.pq")
df_covoting.info()
df_covoting.head()

df_nftsimilarity = pd.read_parquet(data_dir/"num_of_shared_NFT.pq")
df_nftsimilarity.info()
df_nftsimilarity.head()


# %%
df_raw = pd.merge(
        df_dao_voters,
        df_opensea,
        left_on="voter",
        right_on="requestedaddress",
        how="inner",
        validate="m:m",
    ).drop("requestedaddress", axis=1)
df_raw.info()
df_raw.head()


# %% [markdown]
# # Summary statistics of data

# %% [markdown]
# ## distinct N
df_N = (
    df_raw.loc[:, ["dao", "voter", "slug"]]
    .nunique()
    .to_frame("distinct N")
    .T
)
df_N.columns = ["DAO", "voter", "NFT"]
df_N.head()
df_N.pipe(display)


df_voter = df_raw.groupby("voter").agg(
    N_nft_kinds=("slug", "nunique"),
    N_nft_quantity=("owned_asset_count", np.sum),
)
df_voter.head()


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
        log_N_nft_quantity=lambda x: x.log_N_nft_quantity.map(lambda x: f"{x:.2f}"),
    )
    .loc[
        ["mean", "std", "min", "25%", "50%", "75%", "max"],
        ["N_nft_kinds", "log_N_nft_quantity", "N_nft_quantity"],
    ]
)
df_voter_stats.columns = [
    "N(distinct NFT)",
    "log(Total NFT collections)",
    "Total NFT collections",
]
df_voter_stats.pipe(display)


# %% [markdown]
# # Focus on top N NFT collections

# %%
top_N = 20

slug = (
    df_opensea.groupby("slug")
    .owned_asset_count.sum()
    .sort_values(ascending=False)
    .to_frame()
)
slug

slug_top = slug.head(top_N).index.tolist()
df = df_raw.loc[lambda x: x.slug.isin(slug_top)]
df.info()
df.head()


df_voter_N20 = df.groupby("voter").agg(
    N_nft_kinds=("slug", "nunique"),
    N_nft_quantity=("owned_asset_count", np.sum),
)
df_voter_N20.head()


df_voter_N20_stats = (
    df_voter_N20.assign(
        N_nft_quantity_wei=lambda x: x.N_nft_quantity.div(10**18),
        log_N_nft_quantity=lambda x: x.N_nft_quantity.map(np.log),
    )
    .describe()
    .assign(
        N_nft_quantity=lambda x: x.N_nft_quantity.map(lambda x: f"{x:.2e}"),
        N_nft_quantity_wei=lambda x: x.N_nft_quantity_wei.map(lambda x: f"{x:.2e}"),
        N_nft_kinds=lambda x: x.N_nft_kinds.map(lambda x: f"{x:.0f}"),
        log_N_nft_quantity=lambda x: x.log_N_nft_quantity.map(lambda x: f"{x:.2f}"),
    )
    .loc[
        ["mean", "std", "min", "25%", "50%", "75%", "max"],
        ["N_nft_kinds", "log_N_nft_quantity", "N_nft_quantity"],
    ]
)
df_voter_N20_stats.columns = [
    "N(distinct NFT)",
    "log(Total NFT collections)",
    "Total NFT collections",
]
df_voter_N20_stats.pipe(display)

df_N_N20 = (
    df.loc[:, ["dao", "voter", "slug"]]
    .nunique()
    .to_frame("distinct N")
    .T
)
df_N_N20.columns = ["DAO", "voter", "NFT"]
df_N_N20.pipe(display)

# %% [markdown]
# # Data for network visualization
# - Source
# - Target
# - weight


# %%
dir_path = "vis"
!pip install tqdm
!pip install pyvis
from itertools import permutations
from tqdm import tqdm
import networkx as nx
from pyvis.network import Network
import io, pickle


# %% [markdown]
# # Voter network

# %%
network_voters = {}
for grp, df_grp in tqdm(df.groupby("slug")):
    for p in permutations(sorted(df_grp.voter.unique()), 2):
        if p not in network_voters:
            network_voters[p] = 0
        network_voters[p] += 1

# %%
df_network_voter = pd.DataFrame(
    [
        {"Source": source, "Target": target, "weight": weight}
        for (source, target), weight in network_voters.items()
    ]
)
df_network_voter.to_csv(f"{dir_path}/vis_network_voter.csv", index=False)

# %%
# network_voters = {}
# for grp, df_grp in tqdm(df.groupby('slug')):
#     for p in permutations(sorted(df_grp.voter.unique()), 2):
#         if p not in network_voters:
#             network_voters[p] = [0, '']
#         network_voters[p][0] += 1
#         network_voters[p][1] = grp

# df_network_voter = pd.DataFrame([{'Source': source, 'Target': target, 'weight': weight, 'slug': slug} for (source, target), (weight, slug) in network_voters.items()])

# # add colors
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# color_map = dict(zip(df_network_voter.slug.unique(), colors))
# df_network_voter['color'] = df_network_voter.slug.map(lambda x: color_map[x])

# df_network_voter.to_csv(f'{dir_path}/vis_network_voter.csv', index = False)

# %%
df_network_voter.info()
df_network_voter.head()
df_network_voter.weight.value_counts().to_frame("counts")


# %%
df_network_voter[["Source", "Target"]] = df_network_voter[
    ["Source", "Target"]
].applymap(lambda x: x[3:9].lower())

# %%
G_voter = nx.from_pandas_edgelist(
    df_network_voter, source="Source", target="Target", edge_attr=True
)
nx.write_gexf(G_voter,f"{dir_path}/gragh_voter.gexf")
with io.open(f"{dir_path}/gragh_voter.nx", mode="wb") as f:
    pickle.dump(G_voter, f)

# %%
net_voter = Network(height="1200px", width="100%", notebook=True)
net_voter.repulsion()
net_voter.from_nx(G_voter)
net_voter.show(f"{dir_path}/pyvis_nx_voter.html")
