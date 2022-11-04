# %%
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from IPython.display import display
from pprint import pprint
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.DEBUG)

from NFT_categorizer.util import get_openSea_nft, get_prelabeled_nft_category
import config as cfg

# %%
def get_nft_category() -> pd.DataFrame:
    df_category = get_prelabeled_nft_category()
    return df_category

# %%
df_smart_contact = get_openSea_nft(columns=["slug", "primary_asset_contracts"])

# %%
df_category = get_nft_category()


# %%
##############################################################################
sc_opensea = set(df_smart_contact.primary_asset_contracts.unique())
sc_labeled = set(df_category.Smart_contract.unique())
logging.debug(f'%(labeled NFTs): {len(sc_opensea.intersection(sc_labeled)) / len(sc_opensea)}')
##############################################################################

# %%
df_merged = (
    df_smart_contact
    .merge(df_category, how = 'left', left_on = 'primary_asset_contracts', right_on = 'Smart_contract')
)
df_merged.info()

# %%
df_known = (
    df_merged
    .loc[:, ['slug', 'Category']]
    .dropna(subset = ['Category'])

    .drop_duplicates(subset = ['slug', 'Category'], keep = 'first')
    .drop_duplicates(subset = ['slug'], keep = False)
)
df_known.to_csv(cfg.dir_nft_categorizer / "data" / "category_known.csv", index = False)

# %%
df_unknown = (
    df_merged 
    .loc[:, ["slug", "primary_asset_contracts"]]
    .loc[lambda x: ~x.slug.isin(df_known.slug.unique())]
)

df_unknown.to_csv(cfg.dir_nft_categorizer / "data" / "category_unknown.csv", index = False)



