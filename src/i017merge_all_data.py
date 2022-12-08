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

# %%
data_dir = projectfolder / "data"
opensea_downloads = "opensea_collections.pq"
dao_voter_mapping = "dao_voter_mapping.pq"
dao_voters_merged_with_opensea = "dao_voters_merged_with_opensea.pq"

# %%

if not (data_dir / dao_voters_merged_with_opensea).is_file():
    df_opensea = pd.read_parquet(
        data_dir / opensea_downloads,
        columns=[
            "voterid",
            "slug",
            "owned_asset_count",
        ],
    ).drop_duplicates()  # Limit columns, otherwise I'm running out of RAM on the merge
    df_dao_voters = pd.read_parquet(
        data_dir / dao_voter_mapping, columns=["dao", "voterid"]
    ).drop_duplicates()

    merged = pd.merge(
        df_dao_voters,
        df_opensea,
        on="voterid",
        how="inner",
    )
    merged.to_parquet(
        data_dir / dao_voters_merged_with_opensea, index=False, compression="brotli"
    )

else:
    merged = pd.read_parquet(data_dir / dao_voters_merged_with_opensea)
