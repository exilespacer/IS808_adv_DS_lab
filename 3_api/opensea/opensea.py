# %%
import requests
import json
import time
from tqdm import tqdm
from copy import deepcopy


# %%
ENDPOINT = "https://api.opensea.io/api/v1/"


# %%
def get_collections(address):
    """
    Executes GET request to download the first 300 collections of which an address holds at least 1 NFT.
    """
    query = "collections"

    params = {
        "asset_owner": address,
        "offset": 0,
        "limit": 300,
    }

    headers = {"accept": "application/json"}

    response = requests.get(ENDPOINT + query, headers=headers, params=params)

    return json.loads(response.content)


def get_collections_all_users(addresses, sleep_time_in_sec=2):
    """
    Get the collections for a list of users.
    Delay to avoid API issues.
    """
    _coll = []
    for a in tqdm(
        addresses, desc="Querying OpenSea for collections of the provided addresses"
    ):
        r = get_collections(a)
        _coll.append(
            {
                "requestedaddress": a,
                "queryresult": r,
            }
        )

        time.sleep(sleep_time_in_sec)

    return _coll


def extract_attributes(input_data):
    """
    Extracts the interesting attributes from the OpenSea API response for all users in the input data
    """
    _return_dict = {}
    for user in input_data:
        _return_dict["requestedaddress"] = user["requestedaddress"]
        _collections = []
        for collection in user["queryresult"]:
            _attributes = {}

            for attribute in ["slug", "primary_asset_contracts", "owned_asset_count"]:
                if attribute == "primary_asset_contracts":
                    _attributes[attribute] = [
                        x["address"] for x in collection[attribute]
                    ]
                else:
                    _attributes[attribute] = collection[attribute]
            _collections.append(deepcopy(_attributes))
        _return_dict["collections"] = deepcopy(_collections)
    return _return_dict


# %%
rdata = get_collections_all_users(daomemberaddresses)
extract_attributes(rdata)

# %%
