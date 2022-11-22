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
        for vote in json.load(fp):
            # Basic check to only get valid ETH addresses
            if len(vote["voter"]) == 42 and vote["voter"][:2] == "0x":

                # Extract the relevant data
                dao = vote["space"]["id"]
                voter = vote["voter"]

                # Choices can be in different formats
                if isinstance(vote["choice"], dict):
                    choice = vote["choice"]
                elif isinstance(vote["choice"], int) or isinstance(vote["choice"], str):
                    choice = {vote["choice"]: 1}
                elif isinstance(vote["choice"], list):
                    choice = {c: 1 for c in vote["choice"]}
                else:
                    raise ValueError(
                        f"""Unhandled choice type: {type(vote["choice"])} \n {vote["choice"]}"""
                    )

                # Sometimes proposals are None -> Handle this
                if vote["proposal"] is not None:
                    proposalid = vote["proposal"]["id"]
                else:
                    proposalid = None

                # Add new lines for all the choice combinations
                for chc, pwr in choice.items():
                    daolist.append((dao, voter, str(chc), float(pwr), proposalid))


df = pd.DataFrame(daolist, columns=["dao", "voter", "choice", "power", "proposalid"])
del daolist

# %%
# Use parquet for more compression
all_voters = pd.read_parquet(data_dir / "all_voters_with_voterid.pq")

ndf = pd.merge(
    df,
    all_voters,
    how="left",
    left_on="voter",
    right_on="requestedaddress",
    validate="m:1",
).loc[:, ["dao", "voterid", "choice", "power", "proposalid"]]
assert len(df) == len(ndf)
ndf.to_parquet(data_dir / output_file, compression="brotli", index=False)
# %%
