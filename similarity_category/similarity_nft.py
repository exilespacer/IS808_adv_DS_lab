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

import config as cfg 
dir_local_data = cfg.dir_similarity_category / "data"
path_similarity_nft_kinds = cfg.dir_data / "similarity_by_nft_kinds.pq"
path_similarity_nft_distance = cfg.dir_data / "similarity_distance_by_nft.pq"
# input
path_voters = cfg.dir_data / "relevant_voters_with_voterid.pq"
path_category = 'opensea_categories_top50.pq'

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

def compute_similarity_nft_kinds():
    df = load_voter_combinations()
    df['numeric_owned_nft_kinds'] = 0

    df_category = (
        load_voter_NFT_with_category()
        .loc[:, ['voter', 'slug']]
        .dropna()
        .drop_duplicates()
    )

    for slug, df_sub in tqdm(df_category.groupby('slug')):
        voters_in = df_sub.voter.unique()
        increment = (df.voter1.isin(voters_in) * df.voter2.isin(voters_in)).astype(float)
        df[f'numeric_owned_nft_kinds'] += increment
    
    df.to_parquet(path_similarity_nft_kinds, **_parquet_kwargs)
    logging.info('Finish compute_similarity_category')

def compute_similarity_nft_distance():
    df = load_voter_combinations()

    df_nft = (
        load_voter_NFT_with_category()
        .groupby(['voter', 'slug'])
        .shares.sum()
        .unstack(level = 'slug')
    )
    df_category_pct = df_nft.apply(lambda x: x.div(df_nft.sum(axis = 1)), axis = 0)
    nft_dist = dict(zip(df_category_pct.index, df_category_pct.values))

    def get_euclidean_distance(voter1, voter2):
        a = nft_dist[voter1]
        b = nft_dist[voter2]
        distance =  np.linalg.norm(a - b)
        return distance
    
    distance = list(map(lambda x: get_euclidean_distance(*x), df.values))
    df['similarity_nft_distance'] = distance
    
    df.to_parquet(path_similarity_nft_distance, **_parquet_kwargs)
    logging.info(f'Finish {path_similarity_nft_distance}')

def load_similarity_nft_kinds() -> pd.DataFrame:
    df = pd.read_parquet(path_similarity_nft_kinds)
    return df

def load_similarity_nft_distance() -> pd.DataFrame:
    df = pd.read_parquet(path_similarity_nft_distance)
    return df

def main():
    pass

if __name__ == '__main__':
    main()