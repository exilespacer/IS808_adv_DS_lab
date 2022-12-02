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
from math import ceil

from src.i018relevant_voters import relevant_voters_with_voterid

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
temporary_pickle = "_temp.pickle"
shared_daos_between_voters = "shared_daos_between_voters.pq"
# list_of_voters_file = "list_of_voters_used_for_shared_dao_calculations.pq"
binary_similarity = "dao_voters_similarity_binary.pq"
numeric_similarity = "dao_voters_similarity_numeric.pq"
relevant_nft_collections = "opensea_categories_top50.pq"


# %%


def get_input_data(
    in_df: pd.DataFrame,
    relevant_voters=None,
):
    # Reduce the voters to the relevant ones

    df_by_voters = in_df.loc[in_df.index.isin(relevant_voters)]

    # Create the main iteration input dictionary
    lookup_dict = (
        df_by_voters.reset_index()
        .groupby("dao")
        .agg({"voterid": lambda x: set(x)})
        .iloc[:, 0]
        .to_dict()
    )

    return lookup_dict


def create_links(lookup_dict: dict, dump_to_pickle: bool = True):
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
    if dump_to_pickle is True:
        logger.info("Dumping the counter object to pickle")
        with open(data_dir / temporary_pickle, "wb") as f:
            pickle.dump(counter, f)

    return counter


def convert_pickle_to_parquet(_out, columnname="nshareddaos"):
    """
    Loads the pickle created by create_links and converts it to a DataFrame that is stored as parquet
    """

    logger.info("Loading existing pickle into DataFrame")
    with open(data_dir / temporary_pickle, "rb") as f:
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
    (data_dir / temporary_pickle).unlink()

    return df


def generate_or_load_sparse_data(
    sparse_outputfile, list_of_voters: list = [], force: bool = False, **kwargs
):
    """
    Loads / creates the sparse data.
    Pass force=True to recreate all files.
    Pass list_of_voters to limit calculations to these voters.
    """

    if (
        list_of_voters != []
        and force is False
        and (
            (data_dir / temporary_pickle).is_file()
            or (data_dir / sparse_outputfile).is_file()
        )
    ):
        logger.warning(
            "You have provided a list of voters and the files exist, but you are not forcing a complete re-run. Is this what you want? Pass force=True to force."
        )

    # No intermediate or result files exist, or forced -> We need to re-run
    if (
        force is True
        or not (data_dir / temporary_pickle).is_file()
        and not (data_dir / sparse_outputfile).is_file()
    ):

        # Creates the pickle with the results

        df_dao_voters = (
            pd.read_parquet(data_dir / dao_voter_mapping, columns=["dao", "voterid"])
            .drop_duplicates()
            .set_index("voterid")
            .sort_index()
        )
        create_links(
            get_input_data(df_dao_voters, relevant_voters=list_of_voters, **kwargs)
        )

        # Converts everything to parquet
        df = convert_pickle_to_parquet(temporary_pickle, columnname="nshareddaos")

    # Pickle exists -> convert to parquet
    elif (data_dir / temporary_pickle).is_file() and not (
        data_dir / sparse_outputfile
    ).is_file():
        df = convert_pickle_to_parquet(temporary_pickle, columnname="nshareddaos")

    # Parquet exists
    else:
        logger.info("Loading from parquet")
        df = pd.read_parquet(data_dir / sparse_outputfile)

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


def export_dense_dataframes(
    indf=None,
    numeric_outputfile=None,
    binary_outputfile=None,
    batch_size: int = 25_000_000,
    **kwargs,
):
    """
    Exports the numeric full (i.e. non-sparse) regression dataframes.
    """

    if indf is None:
        df = (
            generate_or_load_sparse_data(**kwargs)
            .set_index(["voter1", "voter2"])
            .sort_index()
        )
    else:
        df = indf.set_index(["voter1", "voter2"]).sort_index()

    vs = (
        pd.read_parquet(data_dir / relevant_voters_with_voterid)
        .loc[:, "voterid"]
        .to_list()
    )

    logger.info("Starting export to dense parquet.")

    for id, batch in tqdm(
        enumerate(batched(itertools.combinations(vs, 2), batch_size)),
        total=ceil((len(vs) ** 2 / 2 - len(vs)) / batch_size),
    ):

        nframe = pd.DataFrame(
            0,
            index=pd.MultiIndex.from_tuples(batch, names=["voter1", "voter2"]),
            columns=["_temp"],
        ).sort_index()
        nframe = nframe.join(df, how="left", on=["voter1", "voter2"])[
            [df.columns[0]]
        ].fillna(0)

        if id == 0:
            nframe.reset_index().to_parquet(
                data_dir / numeric_outputfile,
                compression="brotli",
                engine="fastparquet",
            )
            (nframe > 0).reset_index().to_parquet(
                data_dir / binary_outputfile, compression="brotli", engine="fastparquet"
            )
        else:
            nframe.reset_index().to_parquet(
                data_dir / numeric_outputfile,
                compression="brotli",
                engine="fastparquet",
                append=True,
            )
            (nframe > 0).reset_index().to_parquet(
                data_dir / binary_outputfile,
                compression="brotli",
                engine="fastparquet",
                append=True,
            )


def main(list_of_voters: list = [], force: bool = False, **kwargs):
    """
    Performs all steps necessary to obtain results.
    Pass force=True to recreate all files.
    Pass list_of_voters to limit calculations to these voters.
    """

    export_dense_dataframes(list_of_voters=list_of_voters, force=force, **kwargs)

    logger.info("Done")


# %%
if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    # Load list of voters from Chia-Yi
    list_of_voters = (
        pd.read_parquet(data_dir / relevant_voters_with_voterid).iloc[:, 0].to_list()
    )

    main(
        list_of_voters=list_of_voters,
        force=True,
        sparse_outputfile=shared_daos_between_voters,
        binary_outputfile=binary_similarity,
        numeric_outputfile=numeric_similarity,
    )
