# %%
from pathlib import Path

# Duplicate and comment out this row
# It should be the top level folder of the repository
# Sven


# Chia-Yi
# projectfolder = Path("xyz/abc/IS808_adv_DS_lab")
projectfolder = Path("/project/IS808_adv_DS_lab")


import pandas as pd
import numpy as np


data_dir = projectfolder / "data"

# %%
unique_voters_file = data_dir / "unique_voters.json"
pd.read_json(unique_voters_file).to_json(
    projectfolder / "downloader_to_share" / "input_splittable.json",
    orient="records",
    lines=True,
)

# %%
df = pd.read_json(unique_voters_file)
# %%
for idx, part in enumerate(np.array_split(df, 15)):
    part.to_json(
        projectfolder
        / "downloader_to_share"
        / f"input_split_{str(idx+1).zfill(3)}.json",
        orient="records",
        lines=True,
    )

# %%

vset = set(
    pd.read_json(data_dir / "test.json", lines=True, orient="records").iloc[:, 0]
)


# %%
