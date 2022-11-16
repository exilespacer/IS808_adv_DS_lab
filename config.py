
from pathlib import Path

# Duplicate and comment out this row
# It should be the top level folder of the repository
# Sven
# projectfolder = Path("/project/IS808_adv_DS_lab")

# Mengnan
# projectfolder = Path("/mypc/IS808_adv_DS_lab")

# Chia-Yi
projectfolder = Path("/Users/chiayiyen/Projects/IS808_adv_DS_lab")

import os
import sys

os.chdir(projectfolder)  # For interactive testing
sys.path.insert(0, "")  # Required for loading modules

dir_data = projectfolder / "data"
dir_nft_categorizer = projectfolder / "NFT_categorizer"