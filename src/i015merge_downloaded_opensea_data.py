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

from src.util import pd_read_json

# %%
data_dir = projectfolder / "data"
results_folder = data_dir / "opensea_downloaded"
output_file = "opensea_collections.pq"
opensea_categories_full_file = "opensea_categories_full.pq"

# %%

coll = []
for file in tqdm(list(results_folder.glob("*.json"))):
    coll.append(
        pd_read_json(file)
        # .assign(fname=file)
    )
df = (
    pd.concat(coll, axis=0)
    .explode("collections")
    .dropna()
    # Convert to float: Otherwise we get overflow errors, because some people hold insane amounts of their own NFTs (10000000000000000000000)
    # E.g. here: https://opensea.io/assets/ethereum/0xdd63a2d8b2add0709f65b0e7afa19bf95d287b1c/1
    # This causes problems when exporting to parquet files
    .assign(
        slug=lambda x: x["collections"].apply(lambda xx: xx["slug"]),
        owned_asset_count=lambda x: x["collections"].apply(
            lambda xx: float(xx["owned_asset_count"])
        ),
        primary_asset_contracts=lambda x: x["collections"].apply(
            lambda xx: xx["primary_asset_contracts"]
        ),
    )
    .drop("collections", axis=1)
    .explode("primary_asset_contracts")
    .reset_index(drop=True)
)

# %%
# Use parquet for more compression
all_voters = pd.read_parquet(data_dir / "all_voters_with_voterid.pq")

ndf = pd.merge(
    df,
    all_voters,
    how="left",
    left_on="requestedaddress",
    right_on="requestedaddress",
    validate="m:1",
).loc[:, ["voterid", "slug", "owned_asset_count", "primary_asset_contracts"]]
assert len(df) == len(ndf)
ndf.to_parquet(data_dir / output_file, compression="brotli", index=False)
