# %%
import requests
import json
import time
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
from pathlib import Path
import re

# %%
ENDPOINT = "https://api.opensea.io/api/v1/"
projectfolder = Path("/project/IS808_adv_DS_lab")


# %%
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
    save_counter = 1 + max(
        [int(*re.findall(r"_(\d{5})", f.stem)) for f in output_dir.iterdir()]
    )

    def save_file():

        nonlocal output_data

        if isinstance(output_data[0], pd.DataFrame):
            out = pd.concat(output_data, axis=0)
            typ = "df"
        else:
            out = deepcopy(output_data)
            typ = "dict"

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

        if method == "json" and typ == "df":
            out.to_json(output_dir / filename, orient="records")
        elif method == "json" and typ == "dict":
            with open(output_dir / filename, "w") as outfile:
                json.dump(out, outfile)
        elif method in {"pq", "parquet"}:
            out.to_parquet(output_dir / filename)
        else:
            raise ValueError("Invalid filetype")

    try:
        for idx, addr in enumerate(
            tqdm(
                addresses,
                desc="Querying OpenSea for collections of the provided addresses",
            )
        ):
            r = get_collections(addr, sleep_time_in_sec=sleep_time_in_sec)
            output_data.append(r)

            if output_filename is not None and idx % save_interval == 0 and idx > 0:
                save_file()

            time.sleep(sleep_time_in_sec)
    except Exception as exc:
        print(f"Error {exc.__class__}")
        if output_filename is not None:
            save_file()
    if isinstance(output_data[0], pd.DataFrame):
        rval = pd.concat(output_data, axis=0)
    else:
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
if __name__ == "__main__":
    daomemberaddresses = [
        "0xd00e63c587a9f47fa0600a45b87a111d6ea281ec",
        "0x5f8f048759849064d6bf62dfc23d632584c1459c",
        "0xdd45ca901067d65860b54d4310e37371cd92fb09",
        "0xdd7724eff52c6ec5f2f7649eadd5025d890b3690",
        "0xb9b93bf2db3678b010e71ef701d763aee144e93e",
        "0x078ad5270b0240d8529f600f35a271fc6e2b2bd8",
        "0x4d477f1aabcfc2fc3fc9b802e861c013e0123ad9",
        "0xd03ad690ed8065edfdc1e08197a3ebc71535a7ff",
        "0x0ff9b6ab6ec58ceb6d5ae8a1690dd5a0959ad002",
        "0x119383b0051e920d6161cee971247d625b8d69cb",
        "0x1b9da462d07512fa37021973d853b59debb761dd",
        "0x29ee39be789ada5cbb2051f4bdd70dd735aed7d0",
        "0x344b1e4ac175f16d3ba40a688ca928e3768e275a",
        "0x5244736b3a8f898149ae33f013126a20ce7abc62",
        "0x70ddb5abf21202602b57f4860ee1262a594a0086",
        "0x71213a9c9504e406d3242436a4bf69c6bfe74461",
        "0x79ccedbefbfe6c95570d85e65f8b0ac0d6bd017b",
        "0x896002e29fe4cda28a3ae139b0bf7bac26b33a8c",
        "0x8f2a0c6bdd5c1f61552558d2c0d2afe6d3dc5272",
        "0xaeaf8371bf9df3edb08bdd7d05099d49965fdf59",
        "0x00432772ed25d4eb3c6eb26dc461239b35cf8760",
        "0x0516cf37b67235e07af38ad8e388d0e68089b0f2",
        "0x1b5b4fcedf1252cd92496a2fd5c593b39ac49b01",
        "0x2d5823e8e8b4dfbf599a97566ff2a121cc141d60",
        "0x35e6fc00e3f190a8dfe15faa219368a01028ec14",
        "0x487b4705d624b1ecbdc490fa0d645b7421e7c8a4",
        "0x707d306714ff28560f32bf9dae973bd33cd851c5",
        "0x76ac6ad4e4e7c2e0b4ceeb30745bd53df3a85774",
        "0x770bebe5946907cee4dfe004f1648ac435a9d5bb",
        "0x7a3bdee62dd34fac317ae61cd8b3ba7c27ada145",
        "0x972a8b7d891b88220780421fe4d11f174354ceed",
        "0xbda0136ea391e24a938793972726f8763150c7c3",
        "0xc97370f22ed5ac4c7b24a8e1ca9d81febb3b9457",
        "0x4aef8b6ca98ca20087e6b0827a50868172a32afd",
        "0x787c8be38e968bd0b41eaabe0b69d1c32ea09262",
        "0x9ba6baa919bac9acd901df3bfde848fe006d3cae",
        "0xb9be40bdec31ec3935830acc8bcaad1a431f8bb0",
        "0x449e8424e765a04b6b10e10c13ef941a9bf0d48a",
        "0x13dda63c28bb95fe9822fea43ad5feaa966636a9",
        "0x6ae8ea3d4027dfbfdbc9b2e92f2c3d24997d1faa",
        "0xf54ec034555fd6bd9b0161d13e6608b7591b0ee5",
        "0x6ba6c53c8757c99adc983e84ce7e87f240bf7531",
        "0x8325aff2ccc014468cdf20f2538d8534db7a100f",
        "0x721c7fffe9ac426a028e616677811f755fa99eba",
        "0x7d3bf2cb39f5ceeeca91ecc8201dc01616ad089f",
        "0xd1570080bcb784b4f5e378dfd4cdffdfd6c110f6",
        "0xf3baeab4afa486991845d5dbe31069265da39d57",
        "0x16c5efd522b2fdba7342f13e4dc65681ff674f0d",
        "0x17590d2dbb327292d749e4eb06dba4abafd43b47",
        "0x199d53a6982a6281d0e2169b1c6473e0fb00d741",
    ]

    folder = projectfolder / "data/collections"
    rdata = opensea_collections_for_addresses(
        daomemberaddresses[2:10],
        output_dir=folder,
        output_filename="collections.json",
        save_interval=2,
        clear_on_save=True,
    )


# %%

# %%
