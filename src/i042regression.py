from pathlib import Path
from config import projectfolder

# Duplicate and comment out this row
# It should be the top level folder of the repository
# WRDS
# projectfolder = Path("/scratch/mannheim/sv/IS808_adv_DS_lab")

import os
import sys

os.chdir(projectfolder)  # For interactive testing
sys.path.insert(0, "")  # Required for loading modules

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from stargazer.stargazer import Stargazer
import pandas as pd
from math import log

# from stargazer.stargazer import Stargazer
# from tqdm import tqdm
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

data_dir = projectfolder / "data"

specifications_file = "specifications.json"

regression_frame = "regression_frame_merged.pq"
regression_frame_fe = "regression_frame_fe.pq"

regression_results_folder = data_dir / "regression_results"
regression_metadata_file = "regression_metadata.json"

def run_regression(dep_var, indep_var, with_fe, cov_type="HC3", *args, **kwargs):
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

    specifications = load_specifications()

    new_specifications = []
    for i, specification in enumerate(specifications):

        filename = f"regression_{str(i+offset).zfill(5)}.pickle"
        metadata = {**specification, "filename": filename}
        new_specifications.append(metadata)

    with open(data_dir / specifications_file, "w") as f:
        json.dump(new_specifications, f)


def run_specifications(specifications):
    regression_results_folder.mkdir(parents=True, exist_ok=True)

    try:
        offset = (
            max([int(f.stem[-5:]) for f in regression_results_folder.glob("*.pickle")])
            + 1
        )
    except:
        offset = 0

    for i, specification in enumerate(specifications):
        logger.info(f"{specification}")
        run_specification(specification)

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

def run_specification(specification):
    logger.info(f"{specification}")
    r = run_regression(**specification)
    with open(regression_results_folder / specification["filename"], "wb") as f:
        pickle.dump(r, f)
    del r

def add_filenames():
    regression_results_folder.mkdir(parents=True, exist_ok=True)

    try:
        offset = (
            max([int(f.stem[-5:]) for f in regression_results_folder.glob("*.pickle")])
            + 1
        )
    except:
        offset = 0

    specifications = load_specifications()

    new_specifications = []
    for i, specification in enumerate(specifications):

        filename = f"regression_{str(i+offset).zfill(5)}.pickle"
        metadata = {**specification, "filename": filename}
        new_specifications.append(metadata)

    with open(data_dir / specifications_file, "w") as f:
        json.dump(new_specifications, f)

def main():
    logger.info("Starting")
    specifications = load_specifications()
    run_specifications(specifications)
    logger.info("Done")

def pickle2latex(pickle_obj):
    with open(pickle_obj, "rb") as f:
        regression_model = pickle.load(f)

    endog_names = regression_model.model.endog_names
    exog_names = regression_model.model.exog_names

    with_fe = any([v.startswith("fe_") for v in exog_names])

    endog_names = endog_names.replace("_", "")
    exog_names_rename_dict = {x: x.replace("_", "") for x in exog_names if not x.startswith("fe_")}

    covar_order = [x for x in exog_names if not x.startswith("fe_")]

    stargazer = Stargazer([regression_model])
    stargazer.covariate_order(covar_order)
    stargazer.rename_covariates(exog_names_rename_dict)

    tex_file_name = f"{pickle_obj.stem}_{endog_names}.tex"  # Include directory path if needed
    with open(pickle_obj.parent / tex_file_name, "w") as f:
        f.write(stargazer.render_latex())


if __name__ == '__main__':
    # time(SPEC_ID=0 python -m src.i042regression) 
    # main()
    # add_filenames()
    print('make sure that you run add_filenames() in advance')
    spec_id = int(os.environ.get('SPEC_ID', '0'))
    specifications = load_specifications()
    spec = specifications[spec_id]
    print(spec)

    merged = pd.read_parquet(data_dir / regression_frame).set_index(["voter1", "voter2"])

    fe = (
        pd.read_parquet(data_dir / regression_frame_fe)
        .set_index(["voter1", "voter2"])
        .astype(int)
    )
    print('finish loading the data')
    run_specification(spec)
