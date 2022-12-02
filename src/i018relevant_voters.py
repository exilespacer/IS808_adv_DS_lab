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

# Gets or creates a logger
import logging

logger = logging.getLogger(__name__)

# set log level
logger.setLevel(logging.WARN)

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
all_voters_with_voterid = "all_voters_with_voterid.pq"
opensea_downloads = "opensea_collections.pq"
dao_voter_mapping = "dao_voter_mapping.pq"

opensea_categories_file = "opensea_categories_top25.pq"

relevant_voters_with_voterid = "relevant_voters_with_voterid.pq"

# %%


def get_relevant_voters(
    list_of_voters: list = [],
    minimum_number_of_votes=0,
    maximum_number_of_nfts=100_000_000,
    minimum_number_of_nfts=0,
    nft_projects=None,
    votes_after_date="2019-01-01",
):
    """
    Returns the input dictionary required for the next steps.
    If no list_of_voters is passed, it will apply some default filters.
    Otherwise it will use the provided list of voters without further limitation.
    """

    df_dao_voters = pd.read_parquet(
        data_dir / dao_voter_mapping,
        columns=["dao", "voterid", "proposalid", "timestamp"],
    ).drop_duplicates(subset=["dao", "voterid", "proposalid"])

    if len(list_of_voters) == 0:

        logger.info(
            f"Min NFT: {minimum_number_of_nfts} Min votes: {minimum_number_of_votes} File provided: {nft_projects is not None}"
        )

        nft_data = pd.read_parquet(
            data_dir / opensea_downloads, columns=["voterid", "slug"]
        ).drop_duplicates()

        voters_with_nfts = nft_data.groupby("voterid").size()
        voters_with_nfts = set(
            voters_with_nfts[
                (
                    (voters_with_nfts > minimum_number_of_nfts)
                    & (voters_with_nfts <= maximum_number_of_nfts)
                )
            ].index
        )

        if (
            nft_projects is not None
            and isinstance(nft_projects, str)
            and nft_projects.endswith("csv")
        ):
            voters_with_specific_nft_projects = set(
                pd.merge(
                    nft_data,
                    pd.read_csv(data_dir / nft_projects),
                    on="slug",
                    how="inner",
                )["voterid"]
            )
        elif (
            nft_projects is not None
            and isinstance(nft_projects, str)
            and nft_projects.endswith("pq")
        ):
            voters_with_specific_nft_projects = set(
                pd.merge(
                    nft_data,
                    pd.read_parquet(data_dir / nft_projects),
                    on="slug",
                    how="inner",
                )["voterid"]
            )
        elif nft_projects is not None:
            raise ValueError("Something provided for nft_project, but unhandled type")

        # Get the number of proposals for which a voter voted
        nproposals = (
            df_dao_voters.query(f"timestamp > '{votes_after_date}'")
            .groupby(["voterid"])
            .size()
        )
        voters_with_enough_votes = set(
            nproposals[((nproposals >= minimum_number_of_votes))].index
        )

        if nft_projects is not None:
            relevant_voters = (
                voters_with_nfts
                & voters_with_enough_votes
                & voters_with_specific_nft_projects
            )
        else:
            relevant_voters = voters_with_nfts & voters_with_enough_votes

    else:
        relevant_voters = set(list_of_voters)

    logger.info(
        f"Number of voters: {len(relevant_voters)} => {len(relevant_voters)**2/2/1e6:.2f} million combinations"
    )

    return relevant_voters


# %%

if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    # Narrow the datset down
    relevant_voters = get_relevant_voters(
        minimum_number_of_votes=20,
        minimum_number_of_nfts=20,
        maximum_number_of_nfts=10_000,
        nft_projects=opensea_categories_file,
        votes_after_date="2022-08-01",
    )
    logger.info(f"{len(relevant_voters)} relevant voters")

    all_voters = pd.read_parquet(data_dir / all_voters_with_voterid).set_index(
        "voterid"
    )
    rv = all_voters[all_voters.index.isin(relevant_voters)].reset_index()
    rv.to_parquet(
        data_dir / relevant_voters_with_voterid, compression="brotli", index=False
    )
