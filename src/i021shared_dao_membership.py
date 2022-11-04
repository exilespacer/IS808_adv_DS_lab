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
dao_voter_mapping = "dao_voter_mapping.pq"

# %%

df_dao_voters = pd.read_parquet(
    data_dir / dao_voter_mapping, columns=["dao", "voter"]
).drop_duplicates()

# %%
