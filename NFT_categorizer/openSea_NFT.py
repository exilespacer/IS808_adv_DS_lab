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
import NFT_categorizer.model_knn as knn
model = knn.load_knn_classifier()

from sklearn.metrics import confusion_matrix, classification_report

import config as cfg
import os
THRESHOLD = float(os.environ.get('THRESHOLD', '0.1'))

def get_nft_category() -> pd.DataFrame:
    df_category = get_prelabeled_nft_category()
    return df_category

def get_path_dashboard():
    path = cfg.dir_nft_categorizer / "data" / "category_dashboard.csv"
    return path

def get_slug_category(df_slug_contract, print_analysis = True):
    df_category_nft = get_nft_category()
    df_category = (
        df_slug_contract
        .merge(df_category_nft, how = 'left', on = 'Smart_contract')
    )
    df_category_slug = (
        df_category
        .loc[:, ['slug', 'Category']]
        .dropna()
        .drop_duplicates()
    )
    if print_analysis:
        slugs = df_category.loc[lambda x: ~x.Category.isna()].slug.unique()
        N = df_slug_contract.shape[0]
        N_missing = df_slug_contract.Smart_contract.isna().sum()
        N_contract = df_category.Smart_contract.nunique()
        N_category_before = df_category_nft.Smart_contract.nunique()
        N_category_after = df_category.loc[lambda x: x.slug.isin(slugs)].Smart_contract.nunique()
        print(f'''
        Obs: {N}
        missing contract: {N_missing} ({100* N_missing/N:.2f} %)
        total contract: {N_contract}
        known category before pre-filing: {N_category_before} ({100*N_category_before/N_contract:.2f} %)
        known category after pre-filing: {N_category_after} ({100*N_category_after/N_contract:.2f} %)
        ''')
    return df_category_slug
    
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

        df_slug_contract = (
            df_smart_contact
            .loc[:, ['slug', 'Smart_contract']]
            .drop_duplicates()
        )
        df_category_slug = get_slug_category(df_slug_contract)
        
        df_owner_portfolio = (
            df_smart_contact
            .merge(df_category_slug, how = 'left', on = 'slug')
            .assign(N_slug = lambda x: x.groupby(['slug', 'owner']).owned_asset_count.transform('sum'))
            
            .sort_values('Category')
            .drop_duplicates(subset=['slug', 'owner'], keep = 'first')
        )
        df_owner_portfolio = (
            df_owner_portfolio
            .assign(is_true_category = lambda x: ~x.Category.isna())
            .assign(Category = lambda x: x.Category.fillna('Unknown'))
        )
        df_owner_portfolio.to_csv(path, index = False)
        
    df_owner_portfolio = (
        pd.read_csv(path)
        .loc[:, ['owner', 'slug', 'Smart_contract', 'Category', 'is_true_category']]
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
    q = df_owner_portfolio.Unknown.pipe(lambda x: x[x > 0]).quantile(threshold_quantile)
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
        .loc[:, ['Smart_contract', 'Category', 'is_true_category', 'Art', 'Collectible', 'Games', 'Metaverse', 'Other', 'Utility', 'Unknown']]
        .groupby(['Smart_contract', 'Category', 'is_true_category'])
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
    X = knn.transform_X_from_dataframe(df_contract_ownership_structure)
    y_pred = knn.predict(model, X)

    df_predict = (
        df_contract_ownership_structure
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
        .loc[lambda x: x.is_true_category]
        .assign(correct = lambda x: x.Category == x.Predicted)
    )
    precision = df.correct.sum() / df.shape[0]
    
    df_stat = pd.DataFrame(
        confusion_matrix(df.Category, df.Predicted, labels = knn.X_columns),
        index = knn.X_columns,
        columns = knn.X_columns,
    )
    logging.info(df_stat.to_string())
    logging.info(f'\n{classification_report(df.Category, df.Predicted)}')
    return precision

def main():
    max_n_iterations = 10
    df_dashboard = get_dashboard()
    threshold_quantile = THRESHOLD
    for i in range(max_n_iterations):
        df_owner_portfolio = get_owner_portfolio(df_dashboard)
        df_representative_portfolio = get_representative_portfolio(df_owner_portfolio, threshold_quantile = threshold_quantile)
        df_owner_portfolio_category = get_owner_portfolio_category(df_dashboard, df_representative_portfolio)
        if not has_Unknown(df_owner_portfolio_category):
            break
        df_contract_ownership_structure = get_contract_ownership_structure(df_owner_portfolio_category)
        df_predicted_category = get_predicted_category(df_contract_ownership_structure)
        correctness = get_precision(df_predicted_category)
        logging.info(f'''Iteration-{i} | correct-{correctness*100:.2f} | threshold-{threshold_quantile} | known-{df_predicted_category.is_true_category.sum()} | unknown-{(df_predicted_category.is_true_category == False).sum()} | N_added-{(df_predicted_category.Category == 'Unknown').sum()}''')    
        if correctness > 0.5:
            # PROBLEMATIC: mostly assign "Other"
            df_dashboard = update_category(df_dashboard, df_predicted_category) 
            threshold_quantile = THRESHOLD
            continue
        threshold_quantile = threshold_quantile * 0.5
    df_dashboard.to_csv(cfg.dir_nft_categorizer / 'data' / 'category_predicted.csv', index = False)

if __name__ == '__main__':
    # time(THRESHOLD=0.1 python -m NFT_categorizer.openSea_NFT) >& NFT_categorizer/openSea_NFT.log
    main()
