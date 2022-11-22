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

sns.set_theme()
from matplotlib import pyplot as plt
from src.i021shared_dao_membership import numeric_similarity as sourcefile
from src.i021shared_dao_membership import list_of_voters_file
from src.i022shared_dao_membership_votechoice import covoting_between_voters_file

# %%
data_dir = projectfolder / "data"
image_dir = projectfolder / "images"


# %%

df = pd.read_parquet(data_dir / sourcefile)

# %%
# Histogram of the number of shared DAOs
col = "nshareddaos"
ax = sns.histplot(df, x=col, bins=50, log_scale=(False, False))
ax.set_xlabel("Number of shared DAOs")
ax.set_ylabel("Number of voter pairs")

for i, bar in enumerate(ax.patches):
    height = bar.get_height()

    if height == 0:
        continue

    ax.text(
        x=bar.get_x()
        + (
            bar.get_width() / 2
        ),  # x-coordinate position of data label, padded to be in the middle of the bar
        y=height + 100,  # y-coordinate position of data label, padded 0.2 above bar
        s="{:.0f}".format(height),  # data label, formatted to ignore decimals
        ha="center",
    )  # sets horizontal alignment (ha) to center

plt.savefig(image_dir / "histogram_shared_daos.png", bbox_inches="tight")
plt.show()
plt.clf()

# %%

# Get the number of possible pairs
total_possible_voter_pairs = (
    (nuniquevoters := len(pd.read_parquet(data_dir / list_of_voters_file))) ** 2
    - nuniquevoters
) / 2

# Get the number of voters that share at least one DAO
n_nonzero_shared_daos = len(df.query("nshareddaos > 0"))

print(
    f"{n_nonzero_shared_daos/total_possible_voter_pairs:.2%} of possible voter pairs actually have a shared DAO."
)

# %%
total_shared_df = (
    df.groupby("voter1")["nshareddaos"]
    .sum()
    .add(df.groupby("voter2")["nshareddaos"].sum(), fill_value=0)
)
ax = sns.histplot(total_shared_df, bins=50, log_scale=(False, False))
ax.set_xlabel("Number of shared DAOs with all other voters")
ax.set_ylabel("Number of voters")

for i, bar in enumerate(ax.patches):
    height = bar.get_height()

    if height == 0:
        continue

    ax.text(
        x=bar.get_x()
        + (
            bar.get_width() / 2
        ),  # x-coordinate position of data label, padded to be in the middle of the bar
        y=height + 2,  # y-coordinate position of data label, padded 0.2 above bar
        s="{:.0f}".format(height),  # data label, formatted to ignore decimals
        ha="center",
    )  # sets horizontal alignment (ha) to center

plt.savefig(image_dir / "histogram_total_shared_daos.png", bbox_inches="tight")
plt.show()
plt.clf()

# %%

df_covoting = pd.read_parquet(data_dir / covoting_between_voters_file)
# %%
# Histogram of the number of shared choices
col = "nsharedchoices"
ax = sns.histplot(df_covoting, x=col, bins=10, log_scale=(False, True))
ax.set_xlabel("Number of shared Proposal-Choices (non-zero)")
ax.set_ylabel("Number of voter pairs")

for i, bar in enumerate(ax.patches):
    height = bar.get_height()

    if height == 0:
        continue

    ax.text(
        x=bar.get_x()
        + (
            bar.get_width() / 2
        ),  # x-coordinate position of data label, padded to be in the middle of the bar
        y=height + 100,  # y-coordinate position of data label, padded 0.2 above bar
        s="{:.0f}".format(height),  # data label, formatted to ignore decimals
        ha="center",
    )  # sets horizontal alignment (ha) to center

plt.savefig(image_dir / "histogram_shared_choices.png", bbox_inches="tight")
plt.show()
plt.clf()

# %%

# Get the number of possible pairs
total_possible_voter_pairs = (
    (nuniquevoters := len(pd.read_parquet(data_dir / list_of_voters_file))) ** 2
    - nuniquevoters
) / 2

# Get the number of voters that share at least one DAO
n_nonzero_shared_daos = len(df_covoting.query(f"{col} > 0"))

print(
    f"{n_nonzero_shared_daos/total_possible_voter_pairs:.2%} of possible voter pairs have covoted."
)
# %%
