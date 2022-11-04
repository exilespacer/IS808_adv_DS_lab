# %%
from pathlib import Path

# Duplicate and comment out this row
# It should be the top level folder of the repository
# Sven
projectfolder = Path("/project/IS808_adv_DS_lab")
# Mengnan
# projectfolder = Path("/mypc/IS808_adv_DS_lab")


# Chia-Yi
# projectfolder = Path("xyz/abc/IS808_adv_DS_lab")

# %%
import os
import sys

os.chdir(projectfolder)  # For interactive testing
sys.path.insert(0, "")  # Required for loading modules


import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm

# %%
data_dir = projectfolder / "data"
dao_voter_mapping = "dao_voter_mapping.pq"
opensea_downloads = "opensea_collections.pq"
# %%
voters_with_nfts = set(
    pd.read_parquet(data_dir / opensea_downloads, columns=["requestedaddress"])
    .drop_duplicates()
    .iloc[:, 0]
)


df_dao_voters = pd.read_parquet(
    data_dir / dao_voter_mapping, columns=["dao", "voter"]
).drop_duplicates()


# %%
df_by_voters = df_dao_voters.set_index("voter").sort_index()
df_by_voters = df_by_voters.loc[df_by_voters.index.isin(voters_with_nfts)]
relevant_voters = sorted(set(df_by_voters.index))

# %%
_coll = {}
for voter, group in tqdm(df_by_voters.groupby("voter")):
    daos = set(group["dao"])

    for othervoter in tqdm(relevant_voters, leave=False):
        if othervoter != voter:
            otherdaos = set(df_by_voters.loc[[othervoter], "dao"])

            n_shared_daos = len(daos & otherdaos)

            key = tuple(sorted([voter, othervoter]))

            if not key in _coll.keys():
                _coll[key] = n_shared_daos


# %%
