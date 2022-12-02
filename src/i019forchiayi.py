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

opensea_collections = "opensea_collections.pq"
opensea_categories_full_file = "opensea_categories_full.pq"

# %%
oc = pd.read_parquet(
    data_dir / opensea_collections, columns=["voterid", "slug"]
).drop_duplicates()
cat = pd.read_parquet(
    data_dir / opensea_categories_full_file, columns=["slug", "category"]
).drop_duplicates()
# %%
pd.merge(oc.head(), cat, how="left", on="slug").to_parquet(
    data_dir / "voter_slug_category.pq", compression="brotli", index=False
)
