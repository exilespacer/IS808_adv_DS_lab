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
from src.i021shared_dao_membership import (
    get_relevant_voters,
)


# %%
data_dir = projectfolder / "data"
all_voters_with_voterid = "all_voters_with_voterid.pq"
relevant_voters_with_voterid = "relevant_voters_with_voterid.pq"

opensea_categories_file = "opensea_categories_top50.pq"

# %%
# Narrow the datset down
relevant_voters = get_relevant_voters(
    minimum_number_of_votes=25,
    minimum_number_of_nfts=10,
    nft_projects=opensea_categories_file,
)
print(len(relevant_voters))

# %%
all_voters = pd.read_parquet(data_dir / all_voters_with_voterid).set_index(
    "requestedaddress"
)
rv = all_voters[all_voters.index.isin(relevant_voters)].reset_index()
rv.to_parquet(
    data_dir / relevant_voters_with_voterid, compression="brotli", index=False
)
