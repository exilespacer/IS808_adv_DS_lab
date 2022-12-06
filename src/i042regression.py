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

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
import pandas as pd
from math import log
from stargazer.stargazer import Stargazer
from tqdm import tqdm
import pickle
import json

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

specifications_file = "specifications.json"

regression_frame = "regression_frame_merged.pq"
regression_frame_fe = "regression_frame_fe.pq"

regression_results_folder = data_dir / "regression_results"
regression_metadata_file = "regression_metadata.json"
# %%


def run_regression(dep_var, indep_var, with_fe, cov_type="HC3"):
    if with_fe is False:
        mod = sm.OLS(
            merged[dep_var],  # y
            ##################################
            # Without fixed effects
            ##################################
            merged[[*indep_var, "const"]],
            hasconst=True,  # We include our own constants (either FE or a separate const)
        )
        res = mod.fit(
            cov_type="HC3",
        )
    else:
        mod = sm.OLS(
            merged[dep_var],  # y
            ##################################
            # With fixed effects
            ##################################
            pd.concat(
                [
                    merged[indep_var],
                    fe,
                ],
                axis=1,
            ),  # X
            hasconst=True,  # We include our own constants (either FE or a separate const)
        )
        res = mod.fit(
            cov_type=cov_type,
        )

    return res


def run_specifications(specifications):
    regression_results_folder.mkdir(parents=True, exist_ok=True)

    try:
        offset = (
            max([int(f.stem[-5:]) for f in regression_results_folder.glob("*.pickle")])
            + 1
        )
    except:
        offset = 0

    for i, specification in enumerate(tqdm(specifications)):
        r = run_regression(**specification)

        filename = f"regression_{str(i+offset).zfill(5)}.pickle"
        metadata = {**specification, "filename": filename}

        with open(regression_results_folder / filename, "wb") as f:
            pickle.dump(r, f)
        del r

        write_metadata = [metadata]

        if (data_dir / regression_metadata_file).is_file():
            with open(data_dir / regression_metadata_file, "r") as f:
                existing_metadata = json.load(f)

            write_metadata += existing_metadata

        with open(data_dir / regression_metadata_file, "w") as f:
            json.dump(write_metadata, f)


def load_regression_model(filename):

    with open(regression_results_folder / filename, "rb") as f:
        regression_model = pickle.load(f)

    return regression_model


def load_metadata():

    with open(data_dir / regression_metadata_file, "r") as f:
        r = json.load(f)

    return r


def load_specifications():

    with open(data_dir / specifications_file, "r") as f:
        r = json.load(f)

    return r


# %%
merged = pd.read_parquet(data_dir / regression_frame).set_index(["voter1", "voter2"])

fe = (
    pd.read_parquet(data_dir / regression_frame_fe)
    .set_index(["voter1", "voter2"])
    .astype(pd.SparseDtype("bool", 0))
)

# %%
pd.read_parquet(data_dir / regression_frame_fe).to_csv("rffe.csv.gz")
pd.read_parquet(data_dir / "regression_frame_merged.pq").to_csv("rfme.csv.gz")
# %%
logger.info("Starting")
specifications = load_specifications()
run_specifications(specifications)
logger.info("Done")

# %%

# %%
Stargazer([load_regression_model(x["filename"]) for x in load_metadata()])

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
sns.heatmap(
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
            "similarity_category_distance",
            "similarity_nft_distance",
            "cosinesimilarity",
            "anysharednft",
            "numberofsharednft",
            "multiplication",
            "summation",
            "maximum",
            "minimum",
            # "const",
        ]
    ].corr()
)
# %%
import seaborn as sns

# %%
merged[
    [
        "multiplication",
        "summation",
        "maximum",
        "minimum",
        # "const",
    ]
].hist()
# %%
