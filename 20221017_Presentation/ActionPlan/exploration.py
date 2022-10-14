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

opensea = importlib.import_module("3_api.opensea.opensea")

projectfolder = Path("/project/IS808_adv_DS_lab")
# %%

folder = projectfolder / "3_api/deepdao/data/daomembers"

with open(folder.parent / "combined_members.json", "r") as f:
    membersbydao = json.load(f)
# %%
# Apply some basic filters for Ethereum addresses
members = set(
    item
    for sublist in membersbydao.values()
    for item in sublist
    if item[:2] == "0x" and len(item) == 42
)
# %%

data_dir = projectfolder / "3_api/opensea/data/collections"
save_filename = "collections.json"

with open(data_dir / save_filename, "r") as f:
    output_data = json.load(f)

existing = set(a["requestedaddress"] for a in output_data)

todo = members - existing

# %%
r = opensea.opensea_collections_for_addresses(
    todo,
    data_dir=data_dir,
    save_filename=save_filename,
    save_interval=10,
    output_data=output_data,
)
# %%

# %%
