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
from src.i021shared_dao_membership import (
    get_relevant_voters,
    relevant_nft_collections,
    create_links,
    convert_pickle_to_parquet,
    export_regression_dataframes,
)

# Gets or creates a logger
import logging

logger = logging.getLogger(__name__)

# set log level
logger.setLevel(logging.WARN)

# define file handler and set formatter
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s : %(levelname)s : %(module)s - %(funcName)s : %(message)s"
)
stream_handler.setFormatter(formatter)

# add handler to logger
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
logger.addHandler(stream_handler)

# %%
data_dir = projectfolder / "data"
dao_voter_mapping = "dao_voter_mapping.pq"

covoting_between_voters_file = "covoting_between_voters.pq"
binary_outputfile = "dao_voters_similarity_votechoice_binary.pq"
numeric_outputfile = "dao_voters_similarity_votechoice_numeric.pq"

covoting_between_voters_normalized_file = "covoting_between_voters_normalized.pq"
binary_outputfile_normalized = "dao_voters_similarity_votechoice_normalized_binary.pq"
numeric_outputfile_normalized = "dao_voters_similarity_votechoice_normalized_numeric.pq"


# %%
dao = pd.read_parquet(data_dir / numeric_outputfile).set_index(["voter1", "voter2"])
# %%
voter_subset = sorted(set(dao["voter1"].sample(1000)))[:100]
# %%

dao = dao.loc[voter_subset]


# %%
nft = pd.read_parquet(
    data_dir / "similarity_and_sharedNFT_based_NFTcollection.pq"
).rename(lambda x: x.replace(" ", "").lower(), axis=1)
nft[["voter1", "voter2"]] = nft[["voter1", "voter2"]].apply(
    lambda x: pd.Series(sorted(x)), axis=1
)
nft = nft.sort_values(["voter1", "voter2"]).set_index(["voter1", "voter2"])
nft = nft.loc[nft.index.levels[0].intersection(voter_subset)]

# %%

nftcat = (
    pd.read_parquet(data_dir / "similarity_distance_by_category.pq")
    .set_index(["voter1", "voter2"])
    .loc[voter_subset]
)
# %%
nftcat["similarity_category_distance"].hist(bins=100)

# %%

merged = pd.merge(dao, nftcat, how="left", on=["voter1", "voter2"], validate="1:1")
merged = pd.merge(
    merged, nft, how="left", on=["voter1", "voter2"], validate="1:1"
).fillna(0)

# %%
merged.reset_index().to_parquet(data_dir / "regression_frame.pq")
# %%
