# -*- coding: utf-8 -*-
# %%
import os
import sys

os.chdir("/project/IS808_adv_DS_lab")  # For interactive testing
sys.path.insert(0, "")  # Required for loading modules

import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path
import json
import itertools
import importlib
import networkx as nx

opensea = importlib.import_module("3_api.opensea.opensea")
deepdao = importlib.import_module("3_api.deepdao.deepdao")

projectfolder = Path("/project/IS808_adv_DS_lab")
# %%

deepdaomembersfolder = projectfolder / "3_api/deepdao/data" / "daomembers"

members = deepdao.load_downloaded_daos(deepdaomembersfolder)

# Apply some basic filters for Ethereum addresses
members = (
    pd.json_normalize(members)
    .T.reset_index()
    .explode(0)
    .reset_index(drop=True)
    .set_axis(["dao_id", "member_address"], axis=1)
)
members = members[
    members["member_address"].apply(lambda item: item[:2] == "0x" and len(item) == 42)
]
# %%

openseacollectionsfile = (
    projectfolder / "3_api/opensea/data/collections" / "collections.json"
)

with open(openseacollectionsfile, "r") as f:
    openseacollections = json.load(f)

# %%
G = nx.Graph()
# %%
openseacollections
# %%
os_df = pd.DataFrame(openseacollections).explode("collections").dropna()
os_df = pd.concat(
    [
        os_df["requestedaddress"],
        os_df["collections"].apply(pd.Series)[["slug", "owned_asset_count"]],
    ],
    axis=1,
)

# %%
netw_df = pd.merge(
    members,
    os_df,
    left_on="member_address",
    right_on="requestedaddress",
    how="inner",
    validate="m:m",
).drop("requestedaddress", axis=1)
# %%
netw_df.to_csv(
    projectfolder / "20221017_Presentation/ActionPlan" / "network_df.csv", index=False
)

# %%
