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
import NFT_categorizer.util as model_util
import NFT_categorizer.model_knn as knn
model = knn.load_knn_classifier()

from sklearn.metrics import confusion_matrix

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
    # index     : owner
    # columns   : ['Art', 'Collectible', 'Games', 'Metaverse', 'Other', 'Unknown', 'Utility'] in percentage weighting
    df_owner_portfolio = (
        df_dashboard
        .groupby(['owner', 'Category'])
        .Smart_contract.nunique()
        .unstack(level = 'Category')
    
        .dropna()
        .apply(lambda x: x.div(x.sum()), axis = 1)
    )
    return df_owner_portfolio

def get_representative_portfolio(
    df_owner_portfolio:pd.DataFrame, 
    threshold_quantile:float = 0.1
) -> pd.DataFrame:
    q = df_owner_portfolio.Unknown.quantile(threshold_quantile)
    df = (
        df_owner_portfolio
        .loc[lambda x: x.Unknown < q]
    )
    return df

def get_owner_portfolio_category(
    df_dashboard: pd.DataFrame, 
    df_representative_portfolio: pd.DataFrame,
) -> pd.DataFrame:
    valid_owners = df_representative_portfolio.index.unique()
    df = (
        df_dashboard
        .loc[lambda x: x.owner.isin(valid_owners)]
        .merge(df_representative_portfolio.reset_index(), how = 'left', on = ['owner'])
    )
    return df

def has_Unknown(df_owner_portfolio_category):
    return (df_owner_portfolio_category.Unknown > 0).any()

def get_contract_ownership_structure(df_owner_portfolio_category, ignore_unknown = True):
    df = (
        df_owner_portfolio_category
        .loc[:, ['Smart_contract', 'Category', 'Art', 'Collectible', 'Games', 'Metaverse', 'Other', 'Utility', 'Unknown']]
        .groupby(['Smart_contract', 'Category'])
        .mean()
    )
    if ignore_unknown:
        df = (
            df
            .drop(columns = ['Unknown'])
            .apply(lambda x: x.div(x.sum()), axis = 1)
        )
    return df

def get_predicted_category(df_contract_ownership_structure):
    df_X = (
        df_contract_ownership_structure
        .loc[:, model_util.X_columns]
    )
    
    y_pred = knn.knn_predict(model, df_X.values)

    df_predict = (
        df_X
        .index.to_frame().reset_index(drop = True)
        .assign(Predicted = y_pred)
    )
    return df_predict

def update_category(df_dashboard, df_predicted_category):
    df_new_category = (
        df_predicted_category
        .assign(Updated_category = lambda x: x.Category.replace('Unknown', np.nan).fillna(x.Predicted))
        .loc[:, ['Smart_contract', 'Updated_category']]
    )
    # Problematic: Should check with original category, instead of predicted category
    df = (
        df_dashboard
        .merge(df_new_category, how = 'left', on = ['Smart_contract'])
        .assign(Category = lambda x: x.Category.replace('Unknown', np.nan).fillna(x.Updated_category))
        .assign(Category = lambda x: x.Category.fillna('Unknown'))
        .drop(columns = ['Updated_category'])

    )
    return df

def get_selected_voters(quantile_unknown = 0.25):
    df_dashboard = get_dashboard()
    df_owner_portfolio = get_owner_portfolio(df_dashboard)
    voters = (
        df_owner_portfolio
        .loc[lambda x: x.Unknown < x.Unknown.quantile(quantile_unknown)]
        .reset_index()
        .loc[:, ['owner']]
        .drop_duplicates()
    )
    voters.to_csv(cfg.dir_nft_categorizer / 'data' / 'voter.csv', index = False)
    return voters

def get_precision(df_predicted_category):
    df = (
        df_predicted_category
        .loc[lambda x: x.Category != 'Unknown']
        .assign(correct = lambda x: x.Category == x.Predicted)
    )
    precision = df.correct.sum() / df.shape[0]
    
    df_stat = pd.DataFrame(
        confusion_matrix(df.Category, df.Predicted),
        index = model_util.X_columns,
        columns = model_util.X_columns,
    )
    df_stat.pipe(print)
    return precision

def main():
    max_n_iterations = 10
    df_dashboard = get_dashboard()
    threshold_quantile = 0.1
    for i in range(max_n_iterations):
        df_owner_portfolio = get_owner_portfolio(df_dashboard)
        df_representative_portfolio = get_representative_portfolio(df_owner_portfolio, threshold_quantile = threshold_quantile)
        df_owner_portfolio_category = get_owner_portfolio_category(df_dashboard, df_representative_portfolio)
        if not has_Unknown(df_owner_portfolio_category):
            break
        df_contract_ownership_structure = get_contract_ownership_structure(df_owner_portfolio_category)
        df_predicted_category = get_predicted_category(df_contract_ownership_structure)
        precision = get_precision(df_predicted_category)
        if precision > 0.6:
            # PROBLEMATIC: mostly assign "Other"
            df_dashboard = update_category(df_dashboard, df_predicted_category) 
            threshold_quantile = 0.1
            continue
        threshold_quantile = threshold_quantile * 0.5
        logging.info(f'''
        Iteration {i} | precision ({precision*100:.2f} %) | additional category ({(df_predicted_category.Category == 'Unknown').sum()})
        ''')    
    df_dashboard.to_csv(cfg.dir_nft_categorizer / 'data' / 'category_predicted.csv', index = False)

if __name__ == '__main__':
    main()
