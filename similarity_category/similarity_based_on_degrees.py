import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from IPython.display import display
from pprint import pprint
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.DEBUG)

from itertools import combinations
from more_itertools import chunked
import os

from similarity_category.similarity_nft import compute_similarity_nft_kinds, load_similarity_nft_kinds
import config as cfg 
dir_local_data = cfg.dir_similarity_category / "data"

path_similarity_nft_1st_degree = cfg.dir_data / "similarity_by_nft_kinds_1st_degree.pq"
path_similarity_nft_2nd_degree = cfg.dir_data / "similarity_by_nft_kinds_2nd_degree.pq"

# input
path_voters = cfg.dir_data / "relevant_voters_with_voterid.pq"
path_category = 'opensea_categories_top50.pq'
path_similarity_nft_kinds = cfg.dir_data / "similarity_by_nft_kinds.pq"

_parquet_kwargs = {
    "engine": "fastparquet",
    "compression": 'gzip',
    "index": False,
}

def load_voter_combinations():
    from itertools import combinations
    df_voters = pd.read_parquet(path_voters, columns = ['voterid'])
    df_voter_combinations = pd.DataFrame(list(combinations(df_voters.voterid.tolist(), 2)), columns=['voter1', 'voter2'])
    df_voter_combinations = (
        df_voter_combinations
        .drop_duplicates()
        .astype('category')
    )
    return df_voter_combinations

def load_voter_NFT_with_category():
    from similarity_category.util import get_openSea_nft
    # NFT category
    df_labels = pd.read_parquet(cfg.dir_data / path_category, columns=['slug', 'category'])

    # raw NFT data
    df_smart_contact = (
        get_openSea_nft(columns=["slug", "primary_asset_contracts", "voterid", "owned_asset_count"])
        .rename(columns = {
            'primary_asset_contracts': 'smart_contract',
            'voterid': 'voter',
            'owned_asset_count': 'shares',
        })
    )

    df_smart_contact = (
        df_smart_contact
        .loc[lambda x: x.slug.isin(df_labels.slug.unique())]
        .merge(df_labels, on = ['slug'], how = 'left', validate = 'm:m')
    )

    df_smart_contact = df_smart_contact.astype({
        'slug': 'category',
        'smart_contract': 'category',
        'voter': 'category',
        'category': 'category',
        'shares': 'int',
    })
    return df_smart_contact

def get_voter_total_N_nft() -> pd.DataFrame:
    df_slug = load_voter_NFT_with_category()
    df = (
        df_slug
        .groupby('voter')
        .slug.nunique()
        .to_frame('total_N').reset_index()
    )
    return df

def compute_nft_1st_degree_similarity() -> pd.DataFrame:
    df_N = get_voter_total_N_nft()
    df = load_similarity_nft_kinds()

    # calculate similarity starting from voter1 (direction 1)
    df = (
        df
        .merge(df_N.rename(columns = {'voter': 'voter1', 'total_N': 'total_N_voter1'}), on = 'voter1', how = 'left')
        .assign(pct_similar1st_voter1 = lambda x: x.numeric_owned_nft_kinds.div(x.total_N_voter1))
    )

    # calculate similarity starting from voter2 (direction 2)
    df = (
        df
        .merge(df_N.rename(columns = {'voter': 'voter2', 'total_N': 'total_N_voter2'}), on = 'voter2', how = 'left')
        .assign(pct_similar1st_voter2 = lambda x: x.numeric_owned_nft_kinds.div(x.total_N_voter2))
    )

    # get average similarity for two direction
    df['pct_similar1st_avg'] = (df.pct_similar_voter1 + df.pct_similar_voter2) / 2

    df.to_parquet(path_similarity_nft_1st_degree, **_parquet_kwargs)
    logging.info('Finish compute_nft_1st_degree_similarity')

def load_nft_1st_degree_similarity() -> pd.DataFrame:
    df = pd.read_parquet(path_similarity_nft_1st_degree)
    return df

def compute_nft_2nd_degree_similarity() -> pd.DataFrame:
    # TODO
    # df_1st = load_nft_1st_degree_similarity()
    # # create a matrix
    # m_1st_degree = []
    # m_2nd_degree = np.dot(m_1st_degree, m_1st_degree) - 2 * m_1st_degree
    # # assign diagonal to be zero
    # # average directed similarity to be undirected
    pass

def load_nft_2nd_degree_similarity() -> pd.DataFrame:
    df = pd.read_parquet(path_similarity_nft_2nd_degree)
    return df

def main():
    compute_similarity_nft_kinds()
    compute_nft_1st_degree_similarity()
    compute_nft_2nd_degree_similarity()
    pass

if __name__ == '__main__':
    main()



