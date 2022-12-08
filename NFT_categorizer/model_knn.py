#%% packages
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from IPython.display import display
from pprint import pprint
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.DEBUG)

#%% modules
import config as cfg

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# from xgboost import XGBClassifier
from NFT_categorizer.util import get_prelabeled_nft_category

import os
import joblib 

#%% global variables
folder = cfg.dir_nft_categorizer / "data"
X_columns = ['Art', 'Collectible', 'Games', 'Metaverse', 'Other', 'Utility']

#%% IO
def get_path_knn_classifier():
    path = folder/"knn.joblib"
    return path

def load_knn_classifier():
    path = get_path_knn_classifier()
    if not os.path.exists(path):
        raise FileExistsError(f'{path} does not exist.')
    classifier = joblib.load(path)
    return classifier

def dump_knn_classifier(classifier):
    path = get_path_knn_classifier()
    joblib.dump(classifier, path)


#%% prediction
def predict(classifier, X_train):
    X_train = transform_X(X_train)
    y_pred = classifier.predict(X_train)
    return y_pred


#%% X transformation
def transform_X_from_dataframe(df_X):
    X_train = df_X.loc[:, X_columns].values
    return X_train

def transform_X(X_train):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    return X_train

#%% training
def train_knn_classifier(y_train, X_train):
    X_train = transform_X(X_train)

    # Training the K-NN model on the Training set
    classifier = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    return classifier
#%% training process
def get_init_data_y_X(use_cached = True):
    path = folder /'df_contract_char.csv'
    if not use_cached:
        contracts_unique = (
            get_prelabeled_nft_category(only_unique_category=True)
            .Smart_contract
            .dropna()
            .unique()
        )

        df = (
            pd.read_csv(
                folder / "raw/Data_API.csv", 
                usecols=['Datetime_updated', 'Buyer_address', 'Smart_contract', 'Collection_cleaned', 'Category'],
                parse_dates=['Datetime_updated'],)
            .loc[lambda x: x.Smart_contract.isin(contracts_unique)]

            # get the latest category
            .sort_values('Datetime_updated', ascending=False)
            .drop_duplicates(subset = ['Smart_contract', 'Buyer_address'], keep='first')
        )

        df_owners = df.loc[:, ['Buyer_address', 'Smart_contract', 'Category']]

        # get owners collection distribution
        df_owners_portfolio = (
            df
            .groupby('Buyer_address')
            .Category
            .value_counts()

            .unstack(level = 'Category')
            .fillna(0)

            .apply(lambda x: x.div(x.sum()), axis = 1)
            .reset_index()
        )

        df_contract_char = (
            df_owners
            .merge(df_owners_portfolio, how = 'left', on = ['Buyer_address'])
            .groupby(['Category', 'Smart_contract'])
            .mean()
            .reset_index()
        )
        df_contract_char.to_csv(path, index=False)
    
    df_contract_char = pd.read_csv(path)

    df_y = (
        df_contract_char
        .set_index('Smart_contract')
        .loc[:, ['Category']]
    )

    df_X = (
        df_contract_char
        .set_index('Smart_contract')
        .loc[:, X_columns]
    )

    return (df_y, df_X)

def train_initial_knn_classifier(use_cached = True) -> None:
    df_y, df_X = get_init_data_y_X(use_cached = use_cached)
    X = df_X.values
    y = df_y.iloc[:, 0].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    train_knn_classifier(y_train, X_train)

def main():
    # init data from Nature dataset
    df_y, df_X = get_init_data_y_X()
    X = df_X.values
    y = df_y.iloc[:, 0].values

    # data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    # train model
    classifier = train_knn_classifier(y_train, X_train)
    dump_knn_classifier(classifier)

    # evaluation
    classifier = load_knn_classifier()
    y_pred = predict(classifier, X_test)
    labels = sorted(list(set(y_train)))
    df_confusion = (
        pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels), 
        index = labels, columns = labels)
    )
    print(df_confusion)

#%% __main__
if __name__ == '__main__':
    main()