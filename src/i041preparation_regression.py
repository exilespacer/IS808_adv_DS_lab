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
from statsmodels.discrete.discrete_model import Logit
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

covoting_file = "dao_voters_similarity_votechoice_numeric.pq"
coparticipation_file = "dao_voters_similarity_numeric.pq"

relevant_voters_with_voterid = "relevant_voters_with_voterid.pq"

nft_level_similarity = "similarity_and_sharedNFT_based_NFTcollection.pq"
nft_category_similarity = "similarity_by_category.pq"
nft_category_distance = "similarity_distance_by_category.pq"


# %%
voter_subset = sorted(
    pd.read_parquet(data_dir / relevant_voters_with_voterid, columns=["voterid"]).iloc[
        :, 0
    ]
    # .sample(1100)
    .to_list()
)
logger.info(f"{len(voter_subset)} voters.")

# %%
dao_covoting = pd.read_parquet(data_dir / covoting_file).set_index(["voter1", "voter2"])
dao_covoting = dao_covoting.loc[dao_covoting.index.levels[0].intersection(voter_subset)]


# %%
dao_coparticipation = pd.read_parquet(data_dir / coparticipation_file).set_index(
    ["voter1", "voter2"]
)
dao_coparticipation = dao_coparticipation.loc[
    dao_coparticipation.index.levels[0].intersection(voter_subset)
]

# %%
nft = pd.read_parquet(data_dir / nft_level_similarity).rename(
    lambda x: x.replace(" ", "").lower(), axis=1
)
nft[["voter1", "voter2"]] = nft[["voter1", "voter2"]].apply(
    lambda x: pd.Series(sorted(x)), axis=1
)
nft = nft.sort_values(["voter1", "voter2"]).set_index(["voter1", "voter2"])
nft = nft.loc[nft.index.levels[0].intersection(voter_subset)]
nft["anysharednft"] = (nft["numberofsharednft"] > 0).astype(int)

# nft["cosinesimilarity"].hist(bins=100)

# %%

nftcat_euclidiandistance = pd.read_parquet(data_dir / nft_category_distance).set_index(
    ["voter1", "voter2"]
)
nftcat_euclidiandistance = nftcat_euclidiandistance.loc[
    nftcat_euclidiandistance.index.levels[0].intersection(voter_subset)
]
nftcat_euclidiandistance = nftcat_euclidiandistance.divide(
    nftcat_euclidiandistance.max(), axis=1
)

# nftcat_euclidiandistance["similarity_category_distance"].hist(bins=100)

# %%
nftcat_similarity = (
    pd.read_parquet(data_dir / nft_category_similarity)
    .set_index(["voter1", "voter2"])
    .astype(int)
)
nftcat_similarity = nftcat_similarity.loc[
    nftcat_similarity.index.levels[0].intersection(voter_subset)
]

# %%

merged = (
    dao_covoting.merge(
        dao_coparticipation, how="left", on=["voter1", "voter2"], validate="1:1"
    )
    .merge(
        nftcat_euclidiandistance, how="left", on=["voter1", "voter2"], validate="1:1"
    )
    .merge(nftcat_similarity, how="left", on=["voter1", "voter2"], validate="1:1")
    .merge(nft, how="left", on=["voter1", "voter2"], validate="1:1")
    .fillna(0)
)

merged = merged.loc[merged.index.get_level_values(1).isin(voter_subset)]
merged = sm.add_constant(merged, prepend=False)
# del dao, nft, nftcat_euclidiandistance, nftcat_similarity
# %%
# merged.reset_index().to_parquet(data_dir / "regression_frame_merged.pq", compression='brotli',index=False)
merged = (
    pd.read_parquet(data_dir / "regression_frame_merged.pq").set_index(
        ["voter1", "voter2"]
    )
    # .sample(900_000)
)

merged = merged.sample(100_000)
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
    # mod = Logit(
    merged["nshareddaos"],  # y
    pd.concat(
        [
            merged[
                [
                    "similarity_art",
                    "similarity_collectibles",
                    "similarity_domain-names",
                    "similarity_music",
                    "similarity_photography-category",
                    "similarity_sports",
                    "similarity_trading-cards",
                    "similarity_utility",
                    "similarity_virtual-worlds",
                    # "similarity_category_distance",
                    # "const",
                ]
            ],
            fe,
        ],
        axis=1,
    ),  # X
    hasconst=True,  # We include our own constants (either FE or a separate const)
)
res = mod.fit(
    cov_type="HC0",
)
# %%
print(res.summary())

# %%
odf = pd.DataFrame(res.summary().tables[1])
odf.columns = ["variable", *odf.iloc[0, 1:]]
odf = odf.iloc[1:, :]
odf = odf.set_index("variable")
odf = odf.filter(regex="^(?!fe_)", axis=0)
print(odf.to_markdown())
# %%
merged[["similarity_category_distance", "const"]].corr()
# %%
print(pd.DataFrame(res.summary().tables[0]).to_markdown(index=False))
# %%
merged[["similarity_category_distance", "const"]].hist()
# %%
sm.logit
