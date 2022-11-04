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
import dask.dataframe as dd
import itertools
import collections

# %%
data_dir = projectfolder / "data"
dao_voter_mapping = "dao_voter_mapping.pq"
opensea_downloads = "opensea_collections.pq"

minimum_number_of_votes = 10
minimum_number_of_nfts = 10

# %%
voters_with_nfts = (
    pd.read_parquet(data_dir / opensea_downloads, columns=["requestedaddress", "slug"])
    .drop_duplicates()
    .groupby("requestedaddress")
    .size()
)
voters_with_nfts = set(
    voters_with_nfts[voters_with_nfts > minimum_number_of_nfts].index
)

df_dao_voters = pd.read_parquet(
    data_dir / dao_voter_mapping, columns=["dao", "voter", "proposalid"]
).drop_duplicates()

# Get the number of proposals for which a voter voted

nproposals = df_dao_voters.groupby(["voter"]).size()
voters_with_enough_votes = set(nproposals[nproposals >= minimum_number_of_votes].index)

relevant_voters = voters_with_nfts & voters_with_enough_votes

print(
    f"Number of voters: {len(relevant_voters)} => {len(relevant_voters)**2/2/1e9:.2f} billion combinations"
)

df_by_voters = df_dao_voters.drop("proposalid", axis=1).set_index("voter").sort_index()
df_by_voters = df_by_voters.loc[df_by_voters.index.isin(relevant_voters)]

lookup_dict = (
    df_by_voters.groupby("voter").agg({"dao": lambda x: set(x)}).iloc[:, 0].to_dict()
)
# del df_by_voters, voters_with_nfts, voters_with_enough_votes, relevant_voters
# %%
_coll = {}
counter = collections.Counter()
for voter, daos in tqdm(lookup_dict.items()):

    for othervoter in lookup_dict.keys():

        key = tuple(sorted([voter, othervoter]))

        if othervoter != voter and not key in _coll.keys():
            otherdaos = lookup_dict[othervoter]

            n_shared_daos = len(daos & otherdaos)

            _coll[key] = n_shared_daos


# %%
pd.DataFrame(_coll)
# %%
sys.getsizeof(_coll) / 1e6
# %%
len(lookup_dict.keys()) ** 2 / 2 / 1e9
# %%
len(voters_with_nfts), len(voters_with_enough_votes), len(
    voters_with_nfts & voters_with_enough_votes
)

# %%
nproposals[nproposals >= 3]
# %%
nproposals.value_counts().sort_index().iloc[:10].plot.line(logy=False)
# %%
dd.read_parquet(data_dir / opensea_downloads).head()
# %%

# %%
counter = collections.Counter()
# %%
counter.update([("a", "b")])
# %%
