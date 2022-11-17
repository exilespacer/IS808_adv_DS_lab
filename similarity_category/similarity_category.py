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
data_dir = cfg.dir_similarity_category / "data"

_parquet_kwargs = {
    "engine": "fastparquet",
    "compression": 'gzip',
    "index": False,
}

def get_df_category():
    from similarity_category.util import get_openSea_nft
    # NFT category
    df_labels = pd.read_csv(data_dir / "top20_NFT_labeling.csv")

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

    df_category = (
        df_smart_contact
        .groupby(['voter', 'category'])
        .shares.sum()
        .reset_index()
    )
    return df_category

def main(batch_size = 10**4):
    df_category = get_df_category()

    for f in data_dir.glob('*.pq'):
        os.remove(f)

    categories = ('Utility', 'Metaverse', 'Collectible', 'Games', 'Art')

    for category in categories:
        df_sub = df_category.loc[lambda x: x.category == category]
        if df_sub.empty:
            continue

        binary_similarity = f"voters_similarity_category_{category}.pq"
        voters = sorted(df_sub.voter.unique())
        logging.info(f'Start Category - {category}: #(voters) = {len(voters)}')

        # voters = voters[:10**6] #TODO
        voter_combinations = combinations(voters, 2)
        for data in tqdm(chunked(voter_combinations, batch_size)):
            # batch
            df_out = (
                pd.DataFrame(list(data), columns = ['voter1', 'voter2'])
                .astype('category')
                .assign(dummy = True)
            )
            append = True if os.path.exists(data_dir / binary_similarity) else False
            df_out.to_parquet(data_dir / binary_similarity, append = append, **_parquet_kwargs)
        logging.info(f'Finish Category - {category}')
        

if __name__ == '__main__':
    # time(python -m similarity_category.similarity_category) >& similarity_category/similarity_category.log
    main()