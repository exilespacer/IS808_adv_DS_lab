# %%
from pathlib import Path

# Duplicate and comment out this row
# It should be the top level folder of the repository
# Sven
projectfolder = Path("/project/IS808_adv_DS_lab")


# Chia-Yi
# projectfolder = Path("xyz/abc/IS808_adv_DS_lab")

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
opensea_downloads = "collections.pq"
dao_voter_mapping = "dao_voter_mapping.pq"
dao_voters_merged_with_opensea = "dao_voters_merged_with_opensea.pq"

# %%

if not (data_dir / dao_voters_merged_with_opensea).is_file():
    df_opensea = pd.read_parquet(data_dir / opensea_downloads)
    df_dao_voters = pd.read_parquet(data_dir / dao_voter_mapping)

    merged = pd.merge(
        df_dao_voters,
        df_opensea,
        left_on="voter",
        right_on="requestedaddress",
        how="inner",
        validate="m:m",
    ).drop("requestedaddress", axis=1)
    merged.to_parquet(
        data_dir / dao_voters_merged_with_opensea, index=False, compression="brotli"
    )
else:
    merged = pd.read_parquet(data_dir / dao_voters_merged_with_opensea)
# %% [markdown]
# # Summary statistics of data

# %% [markdown]
# ## distinct N

# %%
df_N = (
    df_opensea.loc[:, ["dao_id", "member_address", "slug"]]
    .nunique()
    .to_frame("distinct N")
    .T
)
df_N.columns = ["DAO", "top voter", "NFT"]
df_N

# %%
df_voter = df_opensea.groupby("member_address").agg(
    N_nft_kinds=("slug", "nunique"),
    N_nft_quantity=("owned_asset_count", np.sum),
)
df_voter.head()

# %%
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
df_voter_stats

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
df = df_opensea.loc[lambda x: x.slug.isin(slug_top)]
df.info()

# %% [markdown]
# # Data for network visualization
# - Source
# - Target
# - weight


# %%
dir_path = "vis"

# %% [markdown]
# # DAO network

# %%
network_dao = {}
for grp, df_grp in tqdm(df.groupby("slug")):
    for p in permutations(sorted(df_grp.dao_name.unique()), 2):
        if p not in network_dao:
            network_dao[p] = 0
        network_dao[p] += 1

# %%
df_network_dao = pd.DataFrame(
    [
        {"Source": source, "Target": target, "weight": weight}
        for (source, target), weight in network_dao.items()
    ]
)
df_network_dao.to_csv(f"{dir_path}/vis_network_dao.csv", index=False)

# %%
df_network_dao.info()
df_network_dao.head()
df_network_dao.weight.value_counts().to_frame("counts")

# %%
G_dao = nx.from_pandas_edgelist(
    df_network_dao, source="Source", target="Target", edge_attr="weight"
)

with io.open(f"{dir_path}/gragh_dao.nx", mode="wb") as f:
    pickle.dump(G_dao, f)

# %%
net_dao = Network(height="1200px", width="100%", notebook=True)
net_dao.repulsion()
net_dao.from_nx(G_dao)
net_dao.show(f"{dir_path}/pyvis_nx_dao.html")

# %% [markdown]
# # Voter network

# %%
network_voters = {}
for grp, df_grp in tqdm(df.groupby("slug")):
    for p in permutations(sorted(df_grp.member_address.unique()), 2):
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
#     for p in permutations(sorted(df_grp.member_address.unique()), 2):
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

with io.open(f"{dir_path}/gragh_voter.nx", mode="wb") as f:
    pickle.dump(G_voter, f)

# %%
net_voter = Network(height="1200px", width="100%", notebook=True)
net_voter.repulsion()
net_voter.from_nx(G_voter)
net_voter.show(f"{dir_path}/pyvis_nx_voter.html")

# %% [markdown]
# # additional data

# %%
df_nft = pd.read_csv("data/Data_API.csv")
df_nft.info()

# %%
(df_nft.describe().round(2))

# %%
df_data_nunique = df_nft.loc[
    :,
    [
        "Smart_contract",
        "ID_token",
        "Transaction_hash",
        "Seller_address",
        "Seller_username",
        "Buyer_address",
        "Buyer_username",
        "Image_url_1",
        "Image_url_2",
        "Image_url_3",
        "Image_url_4",
        # 'Price_Crypto',
        "Crypto",
        # 'Price_USD',
        "Name",
        "Description",
        "Collection",
        "Market",
        "Datetime_updated",
        "Datetime_updated_seconds",
        "Permanent_link",
        "Unique_id_collection",
        "Collection_cleaned",
        "Category",
    ],
].nunique()

# %%
df_data_nunique.to_frame("distinct N")

# %%
for i in sorted(df_nft.Category.unique()):
    print(i)

# %%
