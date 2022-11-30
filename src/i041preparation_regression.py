# %%
from pathlib import Path

# Duplicate and comment out this row
# It should be the top level folder of the repository
# Sven
projectfolder = Path("/project/otherprojects/is808")
# Mengnan
# projectfolder = Path("/mypc/IS808_adv_DS_lab")


# Chia-Yi
# projectfolder = Path("xyz/abc/IS808_adv_DS_lab")

# %%
import os
import sys

os.chdir(projectfolder)  # For interactive testing
sys.path.insert(0, "")  # Required for loading modules

import statsmodels.api as sm
import pandas as pd

# Gets or creates a logger
import logging

logger = logging.getLogger(__name__)

# set log level
logger.setLevel(logging.INFO)

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

numeric_outputfile = "dao_voters_similarity_votechoice_numeric.pq"


# %%
voter_subset = sorted(
    pd.read_parquet(data_dir / "relevant_voters_with_voterid.pq", columns=["voterid"])
    .iloc[:, 0]
    .sample(1100)
    .to_list()
)
logger.info(f"{len(voter_subset)} voters.")
# %%
dao = pd.read_parquet(data_dir / numeric_outputfile).set_index(["voter1", "voter2"])
dao = dao.loc[dao.index.levels[0].intersection(voter_subset)]


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

nftcat = pd.read_parquet(data_dir / "similarity_distance_by_category.pq").set_index(
    ["voter1", "voter2"]
)
nftcat = nftcat.loc[nftcat.index.levels[0].intersection(voter_subset)]
# %%
# nftcat["similarity_category_distance"].hist(bins=100)

# %%

merged = pd.merge(dao, nftcat, how="left", on=["voter1", "voter2"], validate="1:1")
merged = (
    pd.merge(merged, nft, how="left", on=["voter1", "voter2"], validate="1:1").fillna(0)
    # .astype(pd.SparseDtype("float", 0))
)
merged = merged.loc[merged.index.get_level_values(1).isin(voter_subset)]
merged = sm.add_constant(merged, prepend=False)
# del dao, nft, nftcat
# %%
merged.reset_index().to_parquet(data_dir / "regression_frame_merged.pq")
merged = (
    pd.read_parquet(data_dir / "regression_frame_merged.pq")
    .set_index(["voter1", "voter2"])
    .sample(900_000)
)

# %%
v1 = pd.get_dummies(
    merged.reset_index()[["voter1"]], columns=["voter1"], sparse=False, prefix="fe"
)
logger.info(f"voter1 FE done")
v2 = pd.get_dummies(
    merged.reset_index()[["voter2"]], columns=["voter2"], sparse=False, prefix="fe"
)
logger.info(f"voter2 FE done")

fe = v1.add(v2, fill_value=0).astype(pd.SparseDtype("bool", 0))
fe.index = merged.index
logger.info(f"Voter FE merged")
del v1, v2

# %%

mod = sm.OLS(
    merged[["nsharedchoices"]],
    pd.concat([merged[["numberofsharednft"]], fe], axis=1),
    hasconst=True,
)
res = mod.fit(
    cov_type="HC0",
)
# %%
print(res.summary())

# %%
res.params
# %%
x = res.get_robustcov_results(cov_type="HC0").summary()
# %%
odf = pd.DataFrame(res.summary().tables[1])
odf.columns = ["variable", *odf.iloc[0, 1:]]
odf = odf.iloc[1:, :]
odf = odf.set_index("variable")
odf = odf.filter(regex="^(?!fe_)", axis=0)
# %%
