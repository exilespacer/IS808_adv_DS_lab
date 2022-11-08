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
# from NFT_categorizer
import config as cfg
import os


def get_nft_category() -> pd.DataFrame:
    df_category = get_prelabeled_nft_category()
    return df_category

def get_path_dashboard():
    path = cfg.dir_nft_categorizer / "data" / "category_dashboard.csv"
    return path

def get_dashboard(used_cached = True) -> pd.DataFrame:    
    path = get_path_dashboard()
    if (not os.path.exists(path)) or (not used_cached):
        df_smart_contact = (
            get_openSea_nft(columns=["slug", "primary_asset_contracts", "requestedaddress", "owned_asset_count"])
            .rename(columns = {
                'primary_asset_contracts': 'Smart_contract',
                'requestedaddress': 'owner'
            })
        )
        df_category = get_nft_category()

        df_owner_portfolio = (
            df_smart_contact
            .merge(df_category, how = 'left', on = 'Smart_contract')
            .assign(N_slug = lambda x: x.groupby(['slug', 'owner']).owned_asset_count.transform('sum'))
            
            .sort_values('Category')
            .drop_duplicates(subset=['slug', 'owner'], keep = 'first')
        )
        df_owner_portfolio = (
            df_owner_portfolio
            .assign(Category = lambda x: x.Category.fillna('Unknown'))
        )
        df_owner_portfolio.to_csv(path, index = False)
        
    df_owner_portfolio = (
        pd.read_csv(path)
        .loc[:, ['owner', 'slug', 'Smart_contract', 'Category']]
        .drop_duplicates()
    )
    return df_owner_portfolio

# shape: (7782732, 4)

# missing:
# - slug            0
# - Smart_contract  561,437
# - owner           0
# - Category        7,079,806

# nunique
# - slug            198,235
# - Smart_contract  69,144
# - owner           453,885
# - Category        6

def get_valid_owner(df_owner_portfolio, threshold_missing = 0.1):
    df_owner_missing = (
        df_owner_portfolio
        .groupby('owner')
        .agg(
            N_total = ('owner', lambda x: len(x)),
            N_missing = ('Category', lambda x: x.isna().sum()),
        )
        .assign(
            pct_missing = lambda x: x.N_missing.div(x.N_total)
        )
        .sort_values('pct_missing')
    )
    print(df_owner_missing.head(10))
    owners_valid = list(df_owner_missing.loc[lambda x: x.pct_missing <= threshold_missing].index.unique())
    return owners_valid

def get_owner_portfolio(df_dashboard):
    df_owner_portfolio = (
        df_dashboard
        .groupby(['owner', 'Category'])
        .Smart_contract.unique()
        .unstack(level = 'Category')
    
        # .apply(lambda x: x.div(x.sum()), axis = 1)
    )
    return df_owner_portfolio


def get_representative_portfolio(df_owner_portfolio, threshold = 0.1):

    # df_existing = df_owner_portfolio 
    # owners_valid = df_existing.pipe(get_valid_owner, threshold_missing = 0.1)
    # df_universe = (
    #     df_existing
    #     .loc[lambda x: x.owner.isin(owners_valid)]
    # )

    pass

def get_owner_portfolio_category(df_dashboard, df_representative_portfolio):
    pass

def has_Unknown(df_owner_portfolio_category):
    pass

def get_contract_ownership_structure(df_owner_portfolio_category):
    pass

def get_predicted_category(df_contract_ownership_structure):
        
    # # TODO
    # X = get_X_train(df_universe)
    # import NFT_categorizer.model_knn as knn

    # classifier = knn.load_knn_classifier()
    # res = zip(X.index, knn.knn_predict(classifier, X.values))
    pass

def update_category(df_dashboard, df_predicted_category):
    pass


def main():
    while True:
        df_dashboard = get_dashboard()
        
        df_owner_portfolio = get_owner_portfolio(df_dashboard)
        df_representative_portfolio = get_representative_portfolio(df_owner_portfolio, threshold = 0.1)
        df_owner_portfolio_category = get_owner_portfolio_category(df_dashboard, df_representative_portfolio)
        if not has_Unknown(df_owner_portfolio_category):
            break
        df_contract_ownership_structure = get_contract_ownership_structure(df_owner_portfolio_category)
        df_predicted_category = get_predicted_category(df_contract_ownership_structure)
        df_dashboard = update_category(df_dashboard, df_predicted_category) 
    
if __name__ == '__main__':
    main()
