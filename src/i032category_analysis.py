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
import seaborn as sns
import collections
import itertools
from math import log

sns.set_theme()
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# %%
data_dir = projectfolder / "data"

opensea_categories_full_file = "opensea_categories_full.pq"

# %%
cat = pd.read_parquet(data_dir / opensea_categories_full_file).drop_duplicates()

# %%
col = "size"
ax = sns.histplot(
    cat.groupby("slug")["category"].agg([col]), x=col, bins=10, log_scale=(False, True)
)
ax.set_xlabel("Number of categories per NFT collection")
ax.set_ylabel("Number of NFT collections")

for i, bar in enumerate(ax.patches):
    height = bar.get_height()

    if height == 0:
        continue

    ax.text(
        x=bar.get_x()
        + (
            bar.get_width() / 2
        ),  # x-coordinate position of data label, padded to be in the middle of the bar
        y=height * 1.1,  # y-coordinate position of data label, padded 0.2 above bar
        s="{:.0f}".format(height),  # data label, formatted to ignore decimals
        ha="center",
    )  # sets horizontal alignment (ha) to center

plt.show()
plt.clf()

# %%


def create_links(lookup_dict):
    counter = collections.Counter()

    # Iterate over all DAOs sorted by size (number of addresses); largest ones first
    for dao in sorted(lookup_dict, key=lambda k: len(lookup_dict[k]), reverse=True):

        # Get the voters for the given DAO
        voters = lookup_dict[dao]

        # Sorted will force the right order for the keys, so that we don't get both A,B and B,A
        for voter, othervoter in itertools.combinations_with_replacement(
            sorted(voters), 2
        ):

            # Add 1 to the voter pair
            counter.update([(voter, othervoter)])
    return counter


category_overlap = create_links(
    cat.groupby("slug").agg({"category": lambda x: set(x)}).iloc[:, 0].to_dict(),
)
# %%
heatmap_df = (
    pd.DataFrame(
        category_overlap.values(),
        index=pd.MultiIndex.from_tuples(category_overlap.keys()),
        columns=["count"],
    )
    .unstack()
    .droplevel(0, axis=1)
    .T
)
# %%
ax = sns.heatmap(heatmap_df, annot=True, fmt=".0f", cmap="crest")

ax.set_title("NFT collections in multiple categories")
plt.show()
plt.clf()
# %%
stacked_bar_df = (
    cat.groupby("category")
    .head(50)
    .pivot(index="category", columns="slug", values="volume_eth")
)
# %%
plt.style.use("dark_background")
ax = stacked_bar_df.sort_values(
    by=stacked_bar_df.index.to_list(), axis=1, ascending=False
).plot(
    kind="bar",
    stacked=True,
    legend=False,
    colormap=ListedColormap(sns.color_palette("viridis", 10)),
    logy=True,
)
ax.set_xlabel("NFT category")
ax.set_ylabel("Traded volume in ETH")
plt.show()
plt.clf()
sns.set_theme()
# %%
