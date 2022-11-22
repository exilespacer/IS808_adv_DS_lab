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
import json


# %%
data_dir = projectfolder / "data"
opensea_categories_har_file = "opensea_rankings.har"
opensea_categories_file = "opensea_categories.pq"

# %%
# To download the HAR file (tested in Chrome)
# Go to https://opensea.io/rankings?category=virtual-worlds&sortBy=total_volume
# Select the entire time horizon
# Open the developer tools in the network tab
# Slowly click through all the categories
# Export everything as "HAR with content"

rows = []
with open(data_dir / opensea_categories_har_file) as f:
    har = json.load(f)
    gqlhar = [
        h
        for h in har["log"]["entries"]
        if h["request"]["url"] == "https://opensea.io/__api/graphql/"
    ]

    for entry in gqlhar:

        nft_list = json.loads(entry["response"]["content"]["text"])["data"]["rankings"][
            "edges"
        ]
        category = json.loads(entry["request"]["postData"]["text"])["variables"][
            "parents"
        ][0]

        for nft in nft_list:
            row = {
                "name": nft["node"]["name"],
                "slug": nft["node"]["slug"],
                "totalquantity": float(nft["node"]["statsV2"]["totalQuantity"]),
                "numowners": float(nft["node"]["windowCollectionStats"]["numOwners"]),
                "numofsales": float(nft["node"]["windowCollectionStats"]["numOfSales"]),
                "volume_eth": float(
                    nft["node"]["windowCollectionStats"]["volume"]["eth"]
                ),
                "category": category,
            }
            rows.append(row)


# %%
df = pd.DataFrame(rows).sort_values(["category", "volume_eth"], ascending=[True, False])
# %%
df.to_parquet(data_dir / opensea_categories_file, compression="brotli", index=False)
