import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


import os
import sys
import config as cfg

os.chdir(cfg.projectfolder)  # For interactive testing
sys.path.insert(0, "")  # Required for loading modules


import pandas as pd
import dask.dataframe as dd


data_dir = cfg.projectfolder / "data"
opensea_downloads = "opensea_collections.pq"
dao_voter_mapping = "dao_voter_mapping.pq"
dao_voters_merged_with_opensea = "dao_voters_merged_with_opensea.pq"
dao_voters_merged_with_opensea_folder = "dao_voters_merged_with_opensea"

nft_prelabeled_category = (
    cfg.projectfolder / "NFT_categorizer" / "data" / "raw" / "Data_API.csv"
)


def get_prelabeled_nft_category(only_unique_category=True):
    df_category = pd.read_csv(
        nft_prelabeled_category,
        usecols=["Datetime_updated", "Smart_contract", "Category"],
        parse_dates=["Datetime_updated"],
    )

    df_category = df_category.sort_values(
        "Datetime_updated", ascending=False
    ).drop_duplicates(subset=["Smart_contract"], keep="first")

    if only_unique_category:
        smart_contracts_with_unique_category = (
            df_category.groupby("Smart_contract")
            .agg(n_category=("Category", lambda x: x.value_counts().shape[0]))
            .loc[lambda x: x.n_category == 1]
            .index.tolist()
        )
        df_category = df_category.loc[
            lambda x: x.Smart_contract.isin(smart_contracts_with_unique_category)
        ]

    df_category = df_category.loc[:, ["Smart_contract", "Category"]].drop_duplicates()
    return df_category


def get_openSea_nft(columns=["voter_id", "slug", "owned_asset_count"], use_dask=False):
    # Limit columns, otherwise I'm running out of RAM on the merge
    if use_dask:
        df_opensea = dd.read_parquet(data_dir / opensea_downloads)[columns].repartition(
            npartitions=25
        )
    else:
        df_opensea = pd.read_parquet(data_dir / opensea_downloads)[columns]
    return df_opensea.drop_duplicates()


def create_merged_opensea_nft() -> None:
    logging.info("Creating merged OpenSea NFT data")
    df_opensea = (
        dd.read_parquet(data_dir / opensea_downloads)[
            [
                "voter_id",
                "slug",
                "owned_asset_count",
            ]  # Limit columns, otherwise I'm running out of RAM on the merge
        ]
        .repartition(npartitions=25)
        .drop_duplicates()
    )
    df_dao_voters = dd.read_parquet(data_dir / dao_voter_mapping)[
        ["dao", "voter"]  # Limit columns, otherwise I'm running out of RAM on the merge
    ].drop_duplicates()

    merged = dd.merge(
        df_dao_voters,
        df_opensea,
        left_on="voter",
        right_on="voter_id",
        how="inner",
    ).drop("voter_id", axis=1)

    merged.to_parquet(
        data_dir / dao_voters_merged_with_opensea_folder,
        write_index=False,
        compression="brotli",
    )
    logging.info("Created merged OpenSea NFT data")


def get_merged_opensea_nft() -> dd.DataFrame:
    if not (data_dir / dao_voters_merged_with_opensea).is_file():
        create_merged_opensea_nft()
    merged = dd.read_parquet(data_dir / dao_voters_merged_with_opensea_folder)
    return merged


if __name__ == "__main__":
    df_merged = get_merged_opensea_nft()
