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
import itertools
import collections
import pickle

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
dao_voter_mapping = "dao_voter_mapping.pq"
opensea_downloads = "opensea_collections.pq"
shared_daos_between_voters_pickle = "shared_daos_between_voters.pickle"
shared_daos_between_voters = "shared_daos_between_voters.pq"
list_of_voters_file = "list_of_voters_used_for_shared_dao_calculations.pq"
binary_similarity = "dao_voters_similarity_binary.pq"
numeric_similarity = "dao_voters_similarity_numeric.pq"
top_20_nfts_list = "top20_NFT_labeling.csv"


# %%


def get_relevant_voters(
    list_of_voters: list = [],
    minimum_number_of_votes=0,
    minimum_number_of_nfts=0,
    nft_projects_csv_file=None,
):
    """
    Returns the input dictionary required for the next steps.
    If no list_of_voters is passed, it will apply some default filters.
    Otherwise it will use the provided list of voters without further limitation.
    """

    df_dao_voters = pd.read_parquet(
        data_dir / dao_voter_mapping, columns=["dao", "voter", "proposalid"]
    ).drop_duplicates()

    if len(list_of_voters) == 0:

        logger.info(
            f"Min NFT: {minimum_number_of_nfts} Min votes: {minimum_number_of_votes} File provided: {nft_projects_csv_file is not None}"
        )

        nft_data = pd.read_parquet(
            data_dir / opensea_downloads, columns=["requestedaddress", "slug"]
        ).drop_duplicates()

        voters_with_nfts = nft_data.groupby("requestedaddress").size()
        voters_with_nfts = set(
            voters_with_nfts[voters_with_nfts > minimum_number_of_nfts].index
        )

        if nft_projects_csv_file is not None:
            voters_with_specific_nft_projects = set(
                pd.merge(
                    nft_data,
                    pd.read_csv(data_dir / nft_projects_csv_file),
                    on="slug",
                    how="inner",
                )["requestedaddress"]
            )

        # Get the number of proposals for which a voter voted
        nproposals = df_dao_voters.groupby(["voter"]).size()
        voters_with_enough_votes = set(
            nproposals[nproposals >= minimum_number_of_votes].index
        )

        if nft_projects_csv_file is not None:
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


def get_input_data(
    in_df: pd.DataFrame,
    relevant_voters=None,
    **kwargs,
):
    if not isinstance(relevant_voters, set):
        relevant_voters = get_relevant_voters(**kwargs)
    else:
        logger.info("Using set of provided relevant voters")

    # Reduce the voters to the relevant ones

    df_by_voters = in_df.loc[df_by_voters.index.isin(relevant_voters)]

    # Create the main iteration input dictionary
    lookup_dict = (
        df_by_voters.reset_index()
        .groupby("dao")
        .agg({"voter": lambda x: set(x)})
        .iloc[:, 0]
        .to_dict()
    )

    # Export list of voters
    pd.DataFrame(sorted(relevant_voters), columns=["voter"]).to_parquet(
        data_dir / list_of_voters_file, compression="brotli", index=False
    )

    return lookup_dict


def create_links(lookup_dict: dict):
    """
    Runs the compute-heavy task of iterating over all DAOs to find common voters.
    Watch the RAM usage!
    """

    logger.info("Rerunning creation")

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
    logger.info("Dumping the counter object to pickle")
    with open(data_dir / shared_daos_between_voters_pickle, "wb") as f:
        pickle.dump(counter, f)

    return counter


def convert_pickle_to_parquet(_in, _out, columnname="nshareddaos"):
    """
    Loads the pickle created by create_links and converts it to a DataFrame that is stored as parquet
    """

    logger.info("Loading existing pickle into DataFrame")
    with open(data_dir / _in, "rb") as f:
        df = pd.DataFrame.from_dict(
            pickle.load(f), orient="index", columns=[columnname]
        ).reset_index()

    # Split up the tuple of voter pairs into separate columns
    df[["voter1", "voter2"]] = pd.DataFrame(df["index"].tolist(), index=df.index)
    # Clean up
    df = df.drop("index", axis=1).loc[:, ["voter1", "voter2", columnname]]

    logger.info("Exporting to parquet")
    df.to_parquet(data_dir / _out, compression="brotli", index=False)
    # Delete the pickle file
    (data_dir / _in).unlink()

    return df


def load_existing_parquet():
    """
    Loads precalculated results from an existing parquet file
    """
    logger.info("Loading from parquet")
    df = pd.read_parquet(data_dir / shared_daos_between_voters)

    return df


# https://docs.python.org/3/library/itertools.html#itertools-recipes
def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


def export_regression_dataframes(batch_size: int = 25_000_000, **kwargs):
    """
    Exports the numeric and binary full (i.e. non-sparse) regression dataframes.
    """

    df = load_data(**kwargs).set_index(["voter1", "voter2"]).sort_index()

    lov = pd.read_parquet(data_dir / list_of_voters_file)

    for id, batch in tqdm(
        enumerate(
            batched(
                itertools.combinations((vs := lov.iloc[:, 0].to_list()), 2), batch_size
            )
        ),
        total=(len(vs) ** 2 / 2 - len(vs)) / batch_size,
    ):

        nframe = pd.DataFrame(
            0,
            index=pd.MultiIndex.from_tuples(batch, names=["voter1", "voter2"]),
            columns=["_temp"],
        ).sort_index()
        nframe = nframe.join(df, how="left", on=["voter1", "voter2"])[
            ["nshareddaos"]
        ].fillna(0)

        if id == 0:
            nframe.reset_index().to_parquet(
                data_dir / numeric_similarity,
                compression="brotli",
                engine="fastparquet",
            )
            (nframe > 0).reset_index().to_parquet(
                data_dir / binary_similarity, compression="brotli", engine="fastparquet"
            )
        else:
            nframe.reset_index().to_parquet(
                data_dir / numeric_similarity,
                compression="brotli",
                engine="fastparquet",
                append=True,
            )
            (nframe > 0).reset_index().to_parquet(
                data_dir / binary_similarity,
                compression="brotli",
                engine="fastparquet",
                append=True,
            )


def load_data(list_of_voters: list = [], force: bool = False, **kwargs):
    """
    Loads / creates the sparse data.
    Pass force=True to recreate all files.
    Pass list_of_voters to limit calculations to these voters.
    """

    if (
        list_of_voters != []
        and force is False
        and (
            (data_dir / shared_daos_between_voters_pickle).is_file()
            or (data_dir / shared_daos_between_voters).is_file()
        )
    ):
        logger.warning(
            "You have provided a list of voters, but are not forcing a re-run. Is this what you want? Pass force=True to force."
        )

    # No intermediate or result files exist, or forced -> We need to re-run
    if (
        force is True
        or not (data_dir / shared_daos_between_voters_pickle).is_file()
        and not (data_dir / shared_daos_between_voters).is_file()
    ):

        # Creates the pickle with the results

        df_dao_voters = (
            pd.read_parquet(data_dir / dao_voter_mapping, columns=["dao", "voter"])
            .drop_duplicates()
            .set_index("voter")
            .sort_index()
        )
        create_links(
            get_input_data(df_dao_voters, list_of_voters=list_of_voters, **kwargs)
        )

        # Converts everything to parquet
        df = convert_pickle_to_parquet(
            shared_daos_between_voters_pickle, shared_daos_between_voters
        )

    # Pickle exists -> convert to parquet
    elif (data_dir / shared_daos_between_voters_pickle).is_file() and not (
        data_dir / shared_daos_between_voters
    ).is_file():
        df = convert_pickle_to_parquet(
            shared_daos_between_voters_pickle, shared_daos_between_voters
        )

    # Parquet exists
    else:
        df = load_existing_parquet()

    return df


def main(list_of_voters: list = [], force: bool = False, **kwargs):
    """
    Performs all steps necessary to obtain results.
    Pass force=True to recreate all files.
    Pass list_of_voters to limit calculations to these voters.
    """

    export_regression_dataframes(list_of_voters=list_of_voters, force=force, **kwargs)

    logger.info("Done")


# %%
if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    # Load list of voters from Chia-Yi
    # list_of_voters = (
    #     pd.read_csv(data_dir / "voter_selected_20221108.csv").iloc[:, 0].to_list()
    # )

    # main(list_of_voters=list_of_voters, force=True)

    main(
        force=True,
        minimum_number_of_votes=25,
        minimum_number_of_nfts=20,
        nft_projects_csv_file=top_20_nfts_list,
    )
