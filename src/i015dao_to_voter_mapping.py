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

import pandas as pd
from tqdm import tqdm
import json

# %%
data_dir = projectfolder / "data"
votes_dir = data_dir / "snapshot_votes"
output_file = "dao_voter_mapping.pq"

# %%


daolist = []
for file in tqdm(list(votes_dir.glob("*/*.json"))):
    with open(file, "r") as fp:
        fc = [
            (vote["space"]["id"], vote["voter"])
            for vote in json.load(fp)
            if len(vote["voter"]) == 42
            and vote["voter"][:2] == "0x"  # Basic check to only get valid ETH addresses
        ]
        daolist.extend(fc)


df = pd.DataFrame(daolist, columns=["dao", "voter"]).drop_duplicates()
del daolist, fc

# %%
# Use parquet for more compression
df.to_parquet(data_dir / output_file, compression="brotli", index=False)
