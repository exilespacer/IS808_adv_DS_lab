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
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from src.i021shared_dao_membership import main as load_data

# %%
data_dir = projectfolder / "data"
image_dir = projectfolder / "20221108_Presentation/images"


# %%

df = load_data()

# %%
# Histogram of the number of shared DAOs
ax = sns.histplot(df, x="nshareddaos", bins=100, log_scale=(False, True))
ax.set_xlabel("Number of shared DAOs")
ax.set_ylabel("Number of voter pairs")
plt.savefig(image_dir / "histogram_shared_daos.png", bbox_inches="tight")

# %%
total_possible_voter_pairs = (
    npossiblecombinations := len(set(df["voter1"]) | set(df["voter2"]))
) ** 2 / 2 - npossiblecombinations
n_nonzero_shared_daos = len(df)
print(
    f"{n_nonzero_shared_daos/total_possible_voter_pairs:.2%} of possible voter pairs actually have a shared DAO."
)
# %%
