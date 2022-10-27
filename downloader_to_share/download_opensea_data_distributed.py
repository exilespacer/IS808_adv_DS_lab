# %%

from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
import json
import time
import re
from copy import deepcopy


# %%

#############################################################
################## FUNCTIONS ################################

ENDPOINT = "https://api.opensea.io/api/v1/"


def get_collections(address, limit=300, sleep_time_in_sec=1):
    """
    Executes GET request to download the all collections of which an address holds at least 1 NFT.
    """
    query = "collections"
    headers = {"accept": "application/json"}

    # 300 is the maximum per request
    n_loaded = limit
    idx = 0

    _coll = []

    while not n_loaded < limit:

        # Sleep only when required
        if idx > 0:
            time.sleep(sleep_time_in_sec)

        params = {
            "asset_owner": address,
            "offset": idx * limit,
            "limit": limit,
        }

        response = requests.get(ENDPOINT + query, headers=headers, params=params)

        js = json.loads(response.content)

        n_loaded = len(js)

        _coll += js
        idx += 1

    extracted = extract_attributes(
        {
            "requestedaddress": address,
            "queryresult": _coll,
        }
    )

    rval = extracted
    # rval = collections_to_dataframe(extracted)

    return rval


def collections_to_dataframe(collectiondata):

    if len(collectiondata["collections"]) == 0:
        return pd.DataFrame()
    else:

        rval = (
            pd.DataFrame([collectiondata])
            .explode("collections")
            .assign(
                slug=lambda x: x["collections"].apply(lambda y: y["slug"]),
                owned_asset_count=lambda x: x["collections"].apply(
                    lambda y: y["owned_asset_count"]
                ),
            )
            .drop("collections", axis=1)
        )

        return rval


def opensea_collections_for_addresses(
    addresses,
    sleep_time_in_sec=2,
    output_dir=None,
    output_filename=None,
    clear_on_save=False,
    save_interval=10,
    output_data=[],
):
    """
    Get the collections for a list of users.
    With delay to avoid API issues.
    """

    # Ensure we don't overwrite existing files
    try:
        save_counter = 1 + max(
            [int(*re.findall(r"_(\d{5})", f.stem)) for f in output_dir.iterdir()]
        )
    except ValueError:
        save_counter = 1

    def save_file():

        nonlocal output_data

        out = deepcopy(output_data)

        method = output_filename.rsplit(".")[-1]

        if clear_on_save:

            nonlocal save_counter

            sv = str(save_counter)
            sv = sv.zfill(5)
            save_counter += 1

            filename = output_filename.replace(f".{method}", f"_{sv}.{method}")

            output_data = []
        else:
            filename = output_filename

        with open(output_dir / filename, "w") as outfile:
            json.dump(out, outfile)

    for idx, addr in enumerate(
        tqdm(
            addresses,
            desc="Querying OpenSea for collections of the provided addresses",
        )
    ):
        try:
            r = get_collections(addr, sleep_time_in_sec=sleep_time_in_sec)
            output_data.append(r)

            if output_filename is not None and idx % save_interval == 0 and idx > 0:
                save_file()

            time.sleep(sleep_time_in_sec)
        except Exception as exc:
            print(f"Error {exc.__class__}")
            if output_filename is not None:
                save_file()
            continue

    rval = output_data
    if output_filename is not None:
        save_file()
    return rval


def extract_attributes(input_data):
    """
    Extracts the interesting attributes from the OpenSea API response
    """
    _collections = []
    for collection in input_data["queryresult"]:
        _attributes = {}

        for attribute in ["slug", "primary_asset_contracts", "owned_asset_count"]:
            if attribute == "primary_asset_contracts":
                _attributes[attribute] = [x["address"] for x in collection[attribute]]
            else:
                _attributes[attribute] = collection[attribute]
        _collections.append(deepcopy(_attributes))
    rval = {
        "requestedaddress": input_data["requestedaddress"],
        "collections": deepcopy(_collections),
    }
    return rval


# %%
#################################################################

# General settings - No need to change anything
output_folder = Path.cwd() / "output_folder"

output_folder.mkdir(parents=True, exist_ok=True)

# Determine the ones that are still left to do
vset = set()
output_filename = 0
for input_file in Path.cwd().glob("*.json"):
    vset |= set(pd.read_json(input_file, lines=True, orient="records").iloc[:, 0])
    if int(input_file.stem[-3:]):
        output_filename = int(input_file.stem[-3:])

cset = set()
for file in output_folder.glob("*.json"):
    with open(file, "r") as f:
        cset |= set(a["requestedaddress"] for a in json.load(f))
todo = vset - cset

# %%

try:
    opensea_collections_for_addresses(
        todo,
        output_dir=output_folder,
        output_filename=f"collections_{str(output_filename).zfill(3)}.json",
        save_interval=100,
        clear_on_save=True,
        sleep_time_in_sec=0.01,  # Don't decrease!
    )
except KeyboardInterrupt:
    print("Exited upon user request.")
