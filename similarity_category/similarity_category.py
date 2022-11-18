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
path_similarity_category = cfg.dir_data / "similarity_by_category.pq"

_parquet_kwargs = {
    "engine": "fastparquet",
    "compression": 'gzip',
    "index": False,
}

def load_voter_combinations():
    path_voter_combinations = cfg.dir_data / "dao_voters_similarity_binary.pq"
    df = (
        pd.read_parquet(path_voter_combinations, columns=['voter1', 'voter2'])
        .drop_duplicates()
    )
    df = df.astype('category')
    return df

def load_voter_NFT_with_category():
    from similarity_category.util import get_openSea_nft
    # NFT category
    df_labels = pd.read_csv(cfg.dir_data / "top20_NFT_labeling.csv")

    # raw NFT data
    df_smart_contact = (
        get_openSea_nft(columns=["slug", "primary_asset_contracts", "requestedaddress", "owned_asset_count"])
        .rename(columns = {
            'primary_asset_contracts': 'smart_contract',
            'requestedaddress': 'voter',
            'owned_asset_count': 'shares',
        })
    )

    df_smart_contact = (
        df_smart_contact
        .loc[lambda x: x.slug.isin(df_labels.slug.unique())]
        .merge(df_labels, on = ['slug'], how = 'left')
    )

    df_smart_contact = df_smart_contact.astype({
        'slug': 'category',
        'smart_contract': 'category',
        'voter': 'category',
        'category': 'category',
        'shares': 'int',
    })
    return df_smart_contact

def compute_similarity_category():
    df = load_voter_combinations()

    df_category = (
        load_voter_NFT_with_category()
        .loc[:, ['voter', 'category']]
        .dropna()
        .drop_duplicates()
    )

    for category, df_sub in df_category.groupby('category'):
        voters_in = df_sub.voter.unique()
        df[f'similarity_{category}'] = df.voter1.isin(voters_in) * df.voter2.isin(voters_in)
    
    df.to_parquet(path_similarity_category, **_parquet_kwargs)
    logging.info('Finish compute_similarity_category')

def load_similarity_category():
    if not os.path.exists(path_similarity_category):
        compute_similarity_category()
    df = pd.read_parquet(path_similarity_category)
    df = df.astype({'voter1': 'category', 'voter2': 'category'})
    return df

def test(batch_size = 10**6):
    voters = list(range(10**5))
    voter_combinations = combinations(voters, 2)
    for data in tqdm(chunked(voter_combinations, batch_size)):
        pass

def create_binary_similarity_category(batch_size = 10**6):
    df_category = load_similarity_category()

    for f in dir_local_data.glob('*.pq'):
        os.remove(f)

    categories = ('Utility', 'Metaverse', 'Collectible', 'Games', 'Art')

    for category in categories:
        df_sub = df_category.loc[lambda x: x.category == category]
        if df_sub.empty:
            continue

        binary_similarity = f"voters_similarity_category_{category}.pq"
        voters = sorted(df_sub.voter.unique())
        logging.info(f'Start Category - {category}: #(voters) = {len(voters)}')

        voter_combinations = combinations(voters, 2)
        for data in tqdm(chunked(voter_combinations, batch_size)):
            df_out = (
                pd.DataFrame(list(data), columns = ['voter1', 'voter2'])
                .astype('category')
                .assign(dummy = True)
            )
            append = True if os.path.exists(dir_local_data / binary_similarity) else False
            df_out.to_parquet(dir_local_data / binary_similarity, append = append, **_parquet_kwargs)
        logging.info(f'Finish Category - {category}')
        
def main():
    df = load_similarity_category()
    df_stats = (
        df
        .set_index(['voter1', 'voter2'])
        .sum().to_frame('N_voter_pairs_coown_this_NFT')
        .assign(pct_voter_pairs_coown_this_NFT = lambda x: 100*x.N_voter_pairs_coown_this_NFT.div(df.shape[0]))
    )
    df_stats.pipe(display)
    df_stats.to_csv(dir_local_data / 'df_stats.csv')
    pass

if __name__ == '__main__':
    main()