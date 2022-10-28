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
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import json
import asyncio

from src.util import gq, gql_all, pd_read_json
from src.opensea import opensea_collections_for_addresses


# %%
gql_folder = projectfolder / "src" / "gql_queries"
data_dir = projectfolder / "data"
votes_dir = data_dir / "snapshot_votes"
filename_spaces = "snapshot_spaces.json"
filename_votes = "votes.json"
unique_voters_file = data_dir / "unique_voters.json"


# %%
def get_spaces(
    data_dir,
    filename,
    force=False,
):
    # Download the data only if we don't have it already
    if not (data_dir / filename).is_file() or force is True:
        spaces_query = gq(gql_folder / "snapshot_spaces.gql")
        # The following command will fail to run in an interactive session, such as Jupyter
        # In that case you need to await gql_all... without asyncio.run()
        asyncio.run(
            gql_all(
                spaces_query,
                field="spaces",
                output_filename=filename,
                output_dir=data_dir,
                batch_size=1000,
            )
        )

    spaces = pd_read_json(data_dir / filename)

    return spaces


spaces = get_spaces(data_dir, filename_spaces, force=False)
# %%

# Make sure we can restart without losing any already done spaces
overall = set(spaces["id"])
done = set(s.name for s in votes_dir.glob("*"))

todo = overall - done

# %%
votes_query = gq(gql_folder / "snapshot_votes_of_space.gql")
for space in tqdm(todo):

    print(f"Space: {space}")
    try:
        asyncio.run(
            gql_all(
                votes_query,
                field="votes",
                save_interval=10,
                clear_on_save=True,
                batch_size=2000,
                rest=False,
                output_filename=filename_votes,
                output_dir=votes_dir / space,
                vars={
                    "space": space,
                },
                counter=False,
            )
        )
    except:
        # Delete all already downloaded files for the space, so that we don't have half-finished downloads in our 'done' list
        shutil.rmtree(votes_dir / space)
        print(f"Deleted {votes_dir / space}")


# %%


vset = set()
for file in votes_dir.glob("*/*.json"):
    with open(file, "r") as fp:
        fc = set(
            v["voter"]
            for v in json.load(fp)
            if len(v["voter"]) == 42 and v["voter"][:2] == "0x"
        )
        vset |= fc

with open(unique_voters_file, "w") as fp:
    json.dump(list(vset), fp)

# %%

# For the next steps see download_data_share.py
