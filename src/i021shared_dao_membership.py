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
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm
import dask.dataframe as dd
import itertools
import collections
import pickle

# %%
data_dir = projectfolder / "data"
dao_voter_mapping = "dao_voter_mapping.pq"
opensea_downloads = "opensea_collections.pq"
shared_daos_between_voters_pickle = "shared_daos_between_voters.pickle"
shared_daos_between_voters = "shared_daos_between_voters.pq"

minimum_number_of_votes = 15
minimum_number_of_nfts = 10

# %%
if not (data_dir / shared_daos_between_voters_pickle).is_file():
    print("Rerunning creation")
    voters_with_nfts = (
        pd.read_parquet(
            data_dir / opensea_downloads, columns=["requestedaddress", "slug"]
        )
        .drop_duplicates()
        .groupby("requestedaddress")
        .size()
    )
    voters_with_nfts = set(
        voters_with_nfts[voters_with_nfts > minimum_number_of_nfts].index
    )

    df_dao_voters = pd.read_parquet(
        data_dir / dao_voter_mapping, columns=["dao", "voter", "proposalid"]
    ).drop_duplicates()

    # Get the number of proposals for which a voter voted

    nproposals = df_dao_voters.groupby(["voter"]).size()
    voters_with_enough_votes = set(
        nproposals[nproposals >= minimum_number_of_votes].index
    )

    relevant_voters = voters_with_nfts & voters_with_enough_votes

    print(
        f"Number of voters: {len(relevant_voters)} => {len(relevant_voters)**2/2/1e6:.2f} million combinations"
    )

    # Reduce the voters to the relevant ones
    df_by_voters = (
        df_dao_voters.drop("proposalid", axis=1).set_index("voter").sort_index()
    )
    df_by_voters = df_by_voters.loc[df_by_voters.index.isin(relevant_voters)]

    # Create the main iteration input dictionary
    lookup_dict = (
        df_by_voters.reset_index()
        .groupby("dao")
        .agg({"voter": lambda x: set(x)})
        .iloc[:, 0]
        .to_dict()
    )
    del (
        df_dao_voters,
        df_by_voters,
        voters_with_nfts,
        voters_with_enough_votes,
        relevant_voters,
    )

    counter = collections.Counter()

    # Iterate over all DAOs sorted by size (number of addresses); largest ones first
    for dao in tqdm(
        sorted(lookup_dict, key=lambda k: len(lookup_dict[k]), reverse=True)
    ):

        # Get the voters for the given DAO
        voters = lookup_dict[dao]

        # Sorted will force the right order for the keys, so that we don't get both A,B and B,A
        for voter, othervoter in itertools.combinations(sorted(voters), 2):

            # Add 1 to the voter pair
            counter.update([(voter, othervoter)])

    # Dump the results into a file to preserve data if running out of memory
    with open(data_dir / shared_daos_between_voters_pickle, "wb") as f:
        pickle.dump(counter, f)
else:
    print("Loading existing pickle")
    with open(data_dir / shared_daos_between_voters_pickle, "rb") as f:
        counter = pickle.load(f)


# %%
if not (data_dir / shared_daos_between_voters).is_file() or True:
    print("Loading into DataFrame")
    df = pd.DataFrame.from_dict(
        counter, orient="index", columns=["nshareddaos"]
    ).reset_index()
    df[["voter1", "voter2"]] = pd.DataFrame(df["index"].tolist(), index=df.index)
    df = df.drop("index", axis=1).loc[:, ["voter1", "voter2", "nshareddaos"]]

    print("Exporting to parquet")
    df.to_parquet(
        data_dir / shared_daos_between_voters, compression="brotli", index=False
    )
    (data_dir / shared_daos_between_voters_pickle).unlink()
# %%
df["nshareddaos"].mean()
# %%
