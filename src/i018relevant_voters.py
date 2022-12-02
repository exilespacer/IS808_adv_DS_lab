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
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

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
    nft_projects=None,
    **kwargs,
):
    """
    Generates the relevant voters given some restrictions
    """

    # Load dataframes from disk
    df_dao_voters = pd.read_parquet(
        data_dir / dao_voter_mapping,
        columns=["dao", "voterid", "proposalid", "timestamp"],
    ).drop_duplicates(subset=["dao", "voterid", "proposalid"])

    nft_data = pd.read_parquet(
        data_dir / opensea_downloads, columns=["voterid", "slug"]
    ).drop_duplicates()
    voters_with_nfts = nft_data.groupby("voterid").size()

    if nft_projects is not None:

        nft_projects_df = pd.read_parquet(data_dir / nft_projects)

    return get_relevant_voters_given_data(
        voters_with_nfts=voters_with_nfts,
        nft_data=nft_data,
        nft_projects_df=nft_projects_df,
        df_dao_voters=df_dao_voters,
        **kwargs,
    )


def get_relevant_voters_given_data(
    minimum_number_of_votes=0,
    maximum_number_of_nfts=100_000_000,
    minimum_number_of_nfts=0,
    votes_after_date="2019-01-01",
    voters_with_nfts=None,
    nft_data=None,
    nft_projects_df=None,
    df_dao_voters=None,
):
    # Information
    logger.info(
        f"NFT: {minimum_number_of_nfts}-{maximum_number_of_nfts} Min votes: {minimum_number_of_votes} File provided: {nft_projects is not None}"
    )

    # Compute the set for number of NFTs
    voters_with_nfts_set = set(
        voters_with_nfts[
            (
                (voters_with_nfts >= minimum_number_of_nfts)
                & (voters_with_nfts <= maximum_number_of_nfts)
            )
        ].index
    )

    # Compute the set for NFT projects
    if nft_projects_df is not None:

        voters_with_specific_nft_projects = set(
            pd.merge(
                nft_data,
                nft_projects_df,
                on="slug",
                how="inner",
            )["voterid"]
        )

    # Get the number of proposals for which a voter voted
    nproposals = (
        df_dao_voters.query(f"timestamp > '{votes_after_date}'")
        .groupby(["voterid"])
        .size()
    )
    voters_with_enough_votes = set(
        nproposals[((nproposals >= minimum_number_of_votes))].index
    )

    # Combine the sets into one set union
    if nft_projects_df is not None:
        relevant_voters = (
            voters_with_nfts_set
            & voters_with_enough_votes
            & voters_with_specific_nft_projects
        )
    else:
        relevant_voters = voters_with_nfts_set & voters_with_enough_votes

    # Information
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

    # %%

    cd = {
        "minimum_number_of_votes": range(1, 40, 3),
        "minimum_number_of_nfts": range(1, 40, 3),
        "maximum_number_of_nfts": range(100, 100_000, 5_000),
        "votes_after_date": [
            "2021-01-01",
            "2022-01-01",
            "2022-05-01",
            "2022-08-01",
        ],
    }

    default = {
        "minimum_number_of_votes": 20,
        "minimum_number_of_nfts": 20,
        "maximum_number_of_nfts": 10000,
        "votes_after_date": "2022-08-01",
    }

    _c = []
    logger.setLevel(logging.WARN)
    for var, values in cd.items():
        for val in tqdm(values):
            vars = {**default, **{var: val}}
            relevant_voters = get_relevant_voters_given_data(
                **vars,
                voters_with_nfts=voters_with_nfts,
                nft_data=nft_data,
                nft_projects_df=nft_projects_df,
                df_dao_voters=df_dao_voters,
            )
            _c.append(
                {
                    **vars,
                    "variable": var,
                    "number of relevant voters": len(relevant_voters),
                }
            )
    df = pd.DataFrame(_c)
    # %%
    def plot_line(indf):
        var = indf.name
        g = sns.lineplot(indf, x=var, y="number of relevant voters")
        g.set(ylim=(0, 20000))
        plt.show()

    df.groupby("variable").apply(plot_line)
