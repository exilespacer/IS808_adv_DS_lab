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
from src.i021shared_dao_membership import (
    create_links,
    convert_pickle_to_parquet,
    export_dense_dataframes,
)

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

covoting_between_voters_file = "covoting_between_voters.pq"
binary_outputfile = "dao_voters_similarity_votechoice_binary.pq"
numeric_outputfile = "dao_voters_similarity_votechoice_numeric.pq"

covoting_between_voters_normalized_file = "covoting_between_voters_normalized.pq"
binary_outputfile_normalized = "dao_voters_similarity_votechoice_normalized_binary.pq"
numeric_outputfile_normalized = "dao_voters_similarity_votechoice_normalized_numeric.pq"


# %%

if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    logger.info("Starting")
    df_dao_voters = (
        pd.read_parquet(
            data_dir / dao_voter_mapping,
            columns=["proposalid", "dao", "choice", "voterid"],
        )
        .set_index("voterid")
        .sort_index()
    )
    relevant_voters = set(
        pd.read_parquet(data_dir / "relevant_voters_with_voterid.pq").loc[:, "voterid"]
    )

    # %%
    ###########################################################################
    ######################## Absolute ########################################
    ###########################################################################
    logger.info("Absolute starting")

    d = (
        df_dao_voters.loc[df_dao_voters.index.isin(relevant_voters)]
        .reset_index()
        .groupby(["proposalid", "dao", "choice"])
        .agg({"voterid": lambda x: set(x)})
        .iloc[:, 0]
        .to_dict()
    )
    lookup_dict = {
        k: v for k, v in d.items() if len(v) >= 2
    }  # It's impossible to have a pair with less than 2 voters
    logger.info("Input data generated, starting link creation")

    # Create links for the different proposals
    links = create_links(lookup_dict)
    convert_pickle_to_parquet(
        covoting_between_voters_file,
        columnname="nsharedchoices",
    )

    export_dense_dataframes(
        indf=pd.read_parquet(data_dir / covoting_between_voters_file),
        binary_outputfile=binary_outputfile,
        numeric_outputfile=numeric_outputfile,
    )
    # %%
    ###########################################################################
    ################### Normalized ###############################
    ###########################################################################
    logger.info("Normalized starting")

    d = (
        df_dao_voters.loc[df_dao_voters.index.isin(relevant_voters)]
        .reset_index()
        .groupby(["dao", "proposalid", "choice"])
        .agg({"voterid": lambda x: set(x)})
        .iloc[:, 0]
        .to_dict()
    )
    lookup_dict = {
        k: v for k, v in d.items() if len(v) >= 2
    }  # It's impossible to have a pair with less than 2 voters

    links = create_links(lookup_dict, dump_to_pickle=False)

    votes_by_voter_dao = (
        df_dao_voters.loc[df_dao_voters.index.isin(relevant_voters)]
        .reset_index()
        # .drop_duplicates(subset=["proposalid", "dao", "voterid"])
        .groupby(["voterid", "dao"])
        .size()
        .to_dict()
    )

    daos_by_voters = (
        df_dao_voters.loc[df_dao_voters.index.isin(relevant_voters)]
        .reset_index()
        # .drop_duplicates(subset=["proposalid", "dao", "voterid"])
        .groupby(["voterid"])
        .agg({"dao": lambda x: set(x)})
        .iloc[:, 0]
        .to_dict()
    )

    normalized_dict = {}
    for (v1, v2), n_shared_choices in tqdm(links.items()):
        shared_daos = daos_by_voters[v1] & daos_by_voters[v2]

        total_votes_in_shared_daos = 0
        v1_total_votes_in_shared_daos = 0
        v2_total_votes_in_shared_daos = 0

        for shared_dao in shared_daos:

            total_votes_in_shared_daos += (
                votes_by_voter_dao[(v1, shared_dao)]
                + votes_by_voter_dao[(v2, shared_dao)]
            )

            v1_total_votes_in_shared_daos += votes_by_voter_dao[(v1, shared_dao)]
            v2_total_votes_in_shared_daos += votes_by_voter_dao[(v2, shared_dao)]

        # Old version
        # normalized_dict[(v1, v2)] = (2 * n_shared_choices) / total_votes_in_shared_daos

        # Jaccard index
        normalized_dict[(v1, v2)] = n_shared_choices / (
            v1_total_votes_in_shared_daos
            + v2_total_votes_in_shared_daos
            - n_shared_choices
        )

    df = pd.DataFrame.from_dict(
        normalized_dict, orient="index", columns=["nsharedchoicesnormalized"]
    ).reset_index()
    # Split up the tuple of voter pairs into separate columns
    df[["voter1", "voter2"]] = pd.DataFrame(df["index"].tolist(), index=df.index)
    # Clean up
    df = df.drop("index", axis=1).loc[
        :, ["voter1", "voter2", "nsharedchoicesnormalized"]
    ]

    df.to_parquet(
        data_dir / covoting_between_voters_normalized_file,
        compression="brotli",
        index=False,
    )

    export_dense_dataframes(
        indf=df,
        binary_outputfile=binary_outputfile_normalized,
        numeric_outputfile=numeric_outputfile_normalized,
    )

    logger.info("Done")
# %%
