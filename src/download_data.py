# %%
from pathlib import Path

# Duplicate and comment out this row
# It should be the top level folder of the repository
# Sven


# Chia-Yi
# projectfolder = Path("xyz/abc/IS808_adv_DS_lab")
projectfolder = Path("/project/IS808_adv_DS_lab")

# %%
import os
import sys

os.chdir(projectfolder)  # For interactive testing
sys.path.insert(0, "")  # Required for loading modules

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import json
import asyncio

from src.util import gq, gql_all, pd_read_json


# %%
gql_folder = projectfolder / "src" / "gql_queries"
data_dir = projectfolder / "data"
filename_spaces = "snapshot_spaces.json"


# %%
# Download the data if we don't have it already
force = False
if not (data_dir / filename_spaces).is_file() or force is True:
    spaces_query = gq(gql_folder / "snapshot_spaces.gql")
    asyncio.run(
        gql_all(
            spaces_query,
            field="spaces",
            output_filename=filename_spaces,
            output_dir=data_dir,
            batch_size=1000,
        )
    )

del force
spaces = pd_read_json(data_dir / filename_spaces)
# %%
