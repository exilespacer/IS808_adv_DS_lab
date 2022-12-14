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
path_similarity_category_distance = cfg.dir_data / "similarity_distance_by_category.pq"
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

def compute_similarity_category_distance():
    df = load_voter_combinations()

    df_category = (
        load_voter_NFT_with_category()
        .groupby(['voter', 'category'])
        .slug.count()
        .unstack(level = 'category')
    )
    df_category_pct = df_category.apply(lambda x: x.div(df_category.sum(axis = 1)), axis = 0)
    category_dist = dict(zip(df_category_pct.index, df_category_pct.values))

    def get_euclidean_distance(voter1, voter2):
        a = category_dist[voter1]
        b = category_dist[voter2]
        distance =  np.linalg.norm(a - b)
        return distance
    
    distance = list(map(lambda x: get_euclidean_distance(*x), df.values))
    df['similarity_category_distance'] = distance
    
    df.to_parquet(path_similarity_category_distance, **_parquet_kwargs)
    logging.info('Finish compute_similarity_category_distance')

def load_similarity_category(use_cached = True):
    if not use_cached:
        os.remove(path_similarity_category)
    if not os.path.exists(path_similarity_category):
        compute_similarity_category()
    df = pd.read_parquet(path_similarity_category)
    df = df.astype({'voter1': 'category', 'voter2': 'category'})
    return df

def load_similarity_category_distance(use_cached = True):
    if not use_cached:
        os.remove(path_similarity_category_distance)
    if not os.path.exists(path_similarity_category_distance):
        compute_similarity_category_distance()
    df = pd.read_parquet(path_similarity_category_distance)
    df = df.astype({'voter1': 'category', 'voter2': 'category'})
    return df

def test(batch_size = 10**6):
    voters = list(range(10**5))
    voter_combinations = combinations(voters, 2)
    for data in tqdm(chunked(voter_combinations, batch_size)):
        pass

def create_binary_similarity_category(batch_size = 10**6):
    # Legacy
    df_category = load_similarity_category()

    for f in dir_local_data.glob('*.pq'):
        os.remove(f)

    categories = ('Utility', 'Metaverse', 'Collectible', 'Games', 'Art')

    for category in categories:
        df_sub = df_category.loc[lambda x: x.category == category]
        if df_sub.empty:
            continue

        path_binary_similarity = f"voters_similarity_category_{category}.pq"
        voters = sorted(df_sub.voter.unique())
        logging.info(f'Start Category - {category}: #(voters) = {len(voters)}')

        voter_combinations = combinations(voters, 2)
        for data in tqdm(chunked(voter_combinations, batch_size)):
            df_out = (
                pd.DataFrame(list(data), columns = ['voter1', 'voter2'])
                .astype('category')
                .assign(dummy = True)
            )
            append = True if os.path.exists(dir_local_data / path_binary_similarity) else False
            df_out.to_parquet(dir_local_data / path_binary_similarity, append = append, **_parquet_kwargs)
        logging.info(f'Finish Category - {category}')

def stats_openSea_category_list(topN = 2):
    df_labels = pd.read_parquet(cfg.dir_data / path_category)
    df = (
        df_labels
        .sort_values('volume_eth', ascending = False)
        .groupby('category')
        .head(topN)
        .loc[:, ['category', 'name', 'volume_eth']]
        .sort_values(['category', 'volume_eth'], ascending = False)
    )
    df.to_csv(dir_local_data / f'category_list_top_{topN}.csv', index = False)
            
def main():
    # binary variable
    df = load_similarity_category(use_cached = False)
    df_stats = (
        df
        .set_index(['voter1', 'voter2'])
        .sum().to_frame('N_voter_pairs_coown_this_NFT')
        .assign(pct_voter_pairs_coown_this_NFT = lambda x: 100*x.N_voter_pairs_coown_this_NFT.div(df.shape[0]))
        .sort_values('pct_voter_pairs_coown_this_NFT', ascending = False)
    )
    df_stats.pipe(display)
    df_stats.to_csv(dir_local_data / 'df_stats_binary.csv')
    

    # Euclidean distance
    df = load_similarity_category_distance(use_cached = False)
    df_stats = (
        df
        .similarity_category_distance
        .describe()
        .to_frame()
    )
    df_stats.pipe(display)
    df_stats.to_csv(dir_local_data / 'df_stats_euclidean_distance.csv')

if __name__ == '__main__':
    main()