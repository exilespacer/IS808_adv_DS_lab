# %%
from pathlib import Path

# Duplicate and comment out this row
# It should be the top level folder of the repository
# Sven
projectfolder = Path("/project/IS808_adv_DS_lab")
# projectfolder = Path("/project/IS808_adv_DS_lab")
# projectfolder = Path("/home/mannheim/svahlpah/notebooks/is808")
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

# %%
data_dir = projectfolder / "data"

regression_frame = "regression_frame_merged.pq"
regression_frame_fe = "regression_frame_fe.pq"

# %%
merged = pd.read_parquet(data_dir / regression_frame).set_index(["voter1", "voter2"])

# %%
# Correlation analysis
merged.groupby(["nsharedcategories"])[["anysharedchoices", "anyshareddaos"]].agg(
    ["mean", "size"]
)
# %%
merged.mean()
# %%
merged[["categorydistance", "collectiondistance"]].corr()
# %%
