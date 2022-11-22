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
import dask.dataframe as dd

# %%
data_dir = projectfolder / "data"
opensea_downloads = "opensea_collections.pq"
dao_voter_mapping = "dao_voter_mapping.pq"
dao_voters_merged_with_opensea = "dao_voters_merged_with_opensea.pq"
dao_voters_merged_with_opensea_folder = "dao_voters_merged_with_opensea"

# %%

if not (data_dir / dao_voters_merged_with_opensea).is_file():
    df_opensea = (
        dd.read_parquet(data_dir / opensea_downloads)[
            [
                "requestedaddress",
                "slug",
                "owned_asset_count",
            ]  # Limit columns, otherwise I'm running out of RAM on the merge
        ]
        .repartition(npartitions=25)
        .drop_duplicates()
    )
    df_dao_voters = dd.read_parquet(data_dir / dao_voter_mapping)[
        ["dao", "voter"]  # Limit columns, otherwise I'm running out of RAM on the merge
    ].drop_duplicates()

    merged = dd.merge(
        df_dao_voters,
        df_opensea,
        left_on="voter",
        right_on="requestedaddress",
        how="inner",
    ).drop("requestedaddress", axis=1)
    merged.to_parquet(
        data_dir / dao_voters_merged_with_opensea_folder,
        write_index=False,
        compression="brotli",
    )

else:
    merged = dd.read_parquet(data_dir / dao_voters_merged_with_opensea_folder)
