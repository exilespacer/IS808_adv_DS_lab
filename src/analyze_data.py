# %%
from pathlib import Path

# Duplicate and comment out this row
# It should be the top level folder of the repository
# Sven


# Chia-Yi
# projectfolder = Path("xyz/abc/IS808_adv_DS_lab")
projectfolder = Path("/project/IS808_adv_DS_lab")

# %%
import os
import sys

os.chdir(projectfolder)  # For interactive testing
sys.path.insert(0, "")  # Required for loading modules

import time

import pandas as pd
from tqdm import tqdm
import json

from src.util import pd_read_json

# %%
results_folder = projectfolder / "downloader_to_share/serverside" / "output_folder"
data_dir = projectfolder / "data"


# %%

coll = []
for file in tqdm(results_folder.glob("*.json")):
    coll.append(pd_read_json(file))
df = (
    pd.concat(coll, axis=0)
    .explode("collections")
    .dropna()
    .assign(
        slug=lambda x: x["collections"].apply(lambda xx: xx["slug"]),
        owned_asset_count=lambda x: x["collections"].apply(
            lambda xx: xx["owned_asset_count"]
        ),
    )
    .drop("collections", axis=1)
)


# %%
