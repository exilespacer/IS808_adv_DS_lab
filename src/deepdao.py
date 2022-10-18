# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import json

projectfolder = Path("/project/IS808_adv_DS_lab")

# %%
# https://martin-thoma.com/configuration-files-in-python/


with open(projectfolder / ".private/keys.json") as keys_file:
    KEYS = json.load(keys_file)

APIKEY = KEYS["DEEPDAO"]
# Note: don't print the key, or if you do, delete the cell's output
# (cell outputs are saved and can be sent to Github).
ENDPOINT = "https://api.deepdao.io/v0.1/"

# %%
def api(query, params=None, post=False):

    # https://datagy.io/python-requests-authentication/
    headers = {"x-api-key": APIKEY, "accept": "application/json"}

    if post:
        response = requests.post(ENDPOINT + query, headers=headers, json=params)
    else:
        response = requests.get(ENDPOINT + query, headers=headers, params=params)

    return response.json()


def get_all_organizations():
    """
    Obtains all the organizationIds from DeepDAO
    """
    query = "organizations"
    res = api(query)
    organizations = [o["organizationId"] for o in res["data"]["resources"]]
    return organizations


def get_top_members_for_dao(organizationId, orderby="votesCount"):
    """
    Obtains the top active members for a DAO
    """

    query = f"organizations/{organizationId}/top_active_in_organization"
    params = {"orderBy": orderby}
    res = api(query, params=params)
    try:
        daomemberaddresses = pd.DataFrame(res["data"])["address"].to_list()
        return daomemberaddresses
    except KeyError:
        return []


def load_downloaded_daos(folder):
    _cdict = {}
    for f in folder.glob("*.json"):
        org = f.stem
        with open(f, "r") as fp:
            data = json.load(fp)

        if len(data) > 0:
            _cdict[org] = data
    return _cdict


# %%


if __name__ == "__main__":
    organizations = set(get_all_organizations())

    folder = projectfolder / "data/daomembers"

    todo = organizations - set(x.stem for x in folder.glob("*.json"))

    for org in tqdm(todo):
        topmembers = get_top_members_for_dao(org)

        with open(folder / f"{org}.json", "w") as f:
            json.dump(topmembers, f)

    # %%
    loaded = load_downloaded_daos(folder)

    with open(folder.parent / f"combined_members.json", "w") as f:
        json.dump(loaded, f)
