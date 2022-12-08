# %%
from pathlib import Path

# Duplicate and comment out this row
# It should be the top level folder of the repository
# Sven
# projectfolder = Path("/project/otherprojects/is808")
projectfolder = Path("/project/IS808_adv_DS_lab/")
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
from math import log

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
covoting_normalized_file = "dao_voters_similarity_votechoice_normalized_numeric.pq"
coparticipation_file = "dao_voters_similarity_numeric.pq"
coparticipation_normalized_file = "dao_voters_similarity_numeric_normalized.pq"

relevant_voters_with_voterid = "relevant_voters_with_voterid.pq"

# Legacy algorithm
# nft_level_similarity = "similarity_and_sharedNFT_based_NFTcollection.pq"
# nft_level_similarity_newmethod = "similarity_secondMethod.pq"

nft_category_similarity_by_individual_categories_file = "similarity_by_category.pq"
nft_category_distance_file = "similarity_distance_by_category.pq"
nft_collection_distance_file = "similarity_distance_by_nft.pq"
nft_collection_distance_first_degree = "similarity_by_nft_kinds_1st_degree.pq"
nft_collection_distance_second_degree = "similarity_by_nft_kinds_2nd_degree.pq"
nft_collection_similarity_by_count_file = "similarity_by_nft_kinds.pq"

regression_frame = "regression_frame_merged.pq"
regression_frame_fe = "regression_frame_fe.pq"

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

dao_covoting["anysharedchoices"] = (dao_covoting["nsharedchoices"] > 0).astype(int)
# %%
dao_covoting_normalized = pd.read_parquet(
    data_dir / covoting_normalized_file
).set_index(["voter1", "voter2"])
dao_covoting_normalized = dao_covoting_normalized.loc[
    dao_covoting_normalized.index.levels[0].intersection(voter_subset)
]

# %%
dao_coparticipation = pd.read_parquet(data_dir / coparticipation_file).set_index(
    ["voter1", "voter2"]
)
dao_coparticipation = dao_coparticipation.loc[
    dao_coparticipation.index.levels[0].intersection(voter_subset)
]

dao_coparticipation["anyshareddaos"] = (dao_coparticipation["nshareddaos"] > 0).astype(
    int
)

# %%
dao_coparticipation_normalized = pd.read_parquet(
    data_dir / coparticipation_normalized_file
).set_index(["voter1", "voter2"])
dao_coparticipation_normalized = dao_coparticipation_normalized.loc[
    dao_coparticipation_normalized.index.levels[0].intersection(voter_subset)
]

# %%
if False:
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
if False:
    nft_newmethod = pd.read_parquet(data_dir / nft_level_similarity_newmethod).rename(
        {"mutliplication": "multiplication"}, axis=1
    )
    nft_newmethod = nft_newmethod.sort_values(["voter1", "voter2"]).set_index(
        ["voter1", "voter2"]
    )
    nft_newmethod = nft_newmethod.loc[
        nft_newmethod.index.levels[0].intersection(voter_subset)
    ]

    nft_newmethod = pd.concat(
        [
            nft_newmethod,
            nft_newmethod.divide(nft_newmethod.max(), axis=1).add_prefix("normalized_"),
        ],
        axis=1,
    )

    # nft["cosinesimilarity"].hist(bins=100)

# %%

nft_category_distance = pd.read_parquet(
    data_dir / nft_category_distance_file
).set_index(["voter1", "voter2"])
nft_category_distance = nft_category_distance.loc[
    nft_category_distance.index.levels[0].intersection(voter_subset)
]
nft_category_distance = nft_category_distance.divide(
    nft_category_distance.max(), axis=1
)

# nft_category_distance["similarity_category_distance"].hist(bins=100)

# %%

nft_collection_similarity_by_count = pd.read_parquet(
    data_dir / nft_collection_similarity_by_count_file
).set_index(["voter1", "voter2"])
nft_collection_similarity_by_count = nft_collection_similarity_by_count.loc[
    nft_collection_similarity_by_count.index.levels[0].intersection(voter_subset)
]

# nft_collection_similarity_by_count["numeric_owned_nft_kinds"].hist(bins=100)

# %%

nft_firstdegree_similarity = (
    pd.read_parquet(data_dir / nft_collection_distance_first_degree)
    .set_index(["voter1", "voter2"])
    .loc[:, ["pct_similar1st_avg"]]
)
nft_firstdegree_similarity = nft_firstdegree_similarity.loc[
    nft_firstdegree_similarity.index.levels[0].intersection(voter_subset)
]

# nft_firstdegree_similarity["pct_similar1st_avg"].hist(bins=100)

# %%

nft_seconddegree_similarity = (
    pd.read_parquet(data_dir / nft_collection_distance_second_degree)
    .set_index(["voter1", "voter2"])
    .loc[:, ["pct_similar2nd_avg"]]
)
nft_seconddegree_similarity = nft_seconddegree_similarity.loc[
    nft_seconddegree_similarity.index.levels[0].intersection(voter_subset)
]

# nft_seconddegree_similarity["pct_similar1st_avg"].hist(bins=100)

# %%
nft_category_similarity_by_individual_categories = (
    pd.read_parquet(data_dir / nft_category_similarity_by_individual_categories_file)
    .set_index(["voter1", "voter2"])
    .astype(int)
)
nft_category_similarity_by_individual_categories = (
    nft_category_similarity_by_individual_categories.loc[
        nft_category_similarity_by_individual_categories.index.levels[0].intersection(
            voter_subset
        )
    ]
)

nft_category_similarity_by_individual_categories[
    "nsharedcategories"
] = nft_category_similarity_by_individual_categories.sum(axis=1)

# %%
nft_collection_distance = pd.read_parquet(
    data_dir / nft_collection_distance_file
).set_index(["voter1", "voter2"])
nft_collection_distance = nft_collection_distance.loc[
    nft_collection_distance.index.levels[0].intersection(voter_subset)
]
nft_collection_distance = nft_collection_distance.divide(
    nft_collection_distance.max(), axis=1
)

# %%

merged = (
    dao_covoting.merge(
        dao_coparticipation, how="left", on=["voter1", "voter2"], validate="1:1"
    )
    .merge(dao_covoting_normalized, how="left", on=["voter1", "voter2"], validate="1:1")
    .merge(
        dao_coparticipation_normalized,
        how="left",
        on=["voter1", "voter2"],
        validate="1:1",
    )
    .merge(nft_category_distance, how="left", on=["voter1", "voter2"], validate="1:1")
    .merge(nft_collection_distance, how="left", on=["voter1", "voter2"], validate="1:1")
    .merge(
        nft_firstdegree_similarity, how="left", on=["voter1", "voter2"], validate="1:1"
    )
    .merge(
        nft_seconddegree_similarity, how="left", on=["voter1", "voter2"], validate="1:1"
    )
    .merge(
        nft_collection_similarity_by_count,
        how="left",
        on=["voter1", "voter2"],
        validate="1:1",
    )
    .merge(
        nft_category_similarity_by_individual_categories,
        how="left",
        on=["voter1", "voter2"],
        validate="1:1",
    )
    # .merge(nft, how="left", on=["voter1", "voter2"], validate="1:1")
    # .merge(nft_newmethod, how="left", on=["voter1", "voter2"], validate="1:1")
    .fillna(0)
)

merged = merged.loc[merged.index.get_level_values(1).isin(voter_subset)]
merged = sm.add_constant(merged, prepend=False)

merged = merged.rename(
    {
        "similarity_category_distance": "categorydistance",
        "similarity_nft_distance": "collectiondistance",
        "pct_similar1st_avg": "firstdegreesimilarity",
        "pct_similar2nd_avg": "seconddegreesimilarity",
        "numeric_owned_nft_kinds": "nsharedcollections",
        "similarity_art": "art",
        "similarity_collectibles": "collectibles",
        "similarity_domain-names": "domainnames",
        "similarity_music": "music",
        "similarity_photography-category": "photographycategory",
        "similarity_sports": "sports",
        "similarity_trading-cards": "tradingcards",
        "similarity_utility": "utility",
        "similarity_virtual-worlds": "virtualworlds",
    },
    axis=1,
)

merged.reset_index().to_parquet(
    data_dir / regression_frame, compression="brotli", index=False
)
# merged.reset_index().to_csv(data_dir / "rfme.csv.gz")

# %%
v1 = pd.get_dummies(
    merged.reset_index()[["voter1"]], columns=["voter1"], sparse=False, prefix="fe"
)
logger.info(f"voter1 FE done")
v2 = pd.get_dummies(
    merged.reset_index()[["voter2"]], columns=["voter2"], sparse=False, prefix="fe"
)
logger.info(f"voter2 FE done")

fe = v1.add(v2, fill_value=0).sort_index(axis=1)
fe.index = merged.index
logger.info(f"Voter FE merged")
del v1, v2

fe.reset_index().to_parquet(
    data_dir / regression_frame_fe, compression="brotli", index=False
)
# fe.reset_index().to_csv(data_dir / "rffe.csv.gz")

# %%
logger.info(f"Done")
