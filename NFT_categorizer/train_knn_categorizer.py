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

import config as cfg
from NFT_categorizer.util import get_openSea_nft

import os
import joblib 
# %%
folder = cfg.dir_nft_categorizer / "data"

def get_data_y_X(use_cached = True):
    path = folder /'df_contract_char.csv'
    if not use_cached:
        df = (
            pd.read_csv(
                folder / "raw/Data_API.csv", 
                usecols=['Datetime_updated', 'Buyer_address', 'Smart_contract', 'Collection_cleaned', 'Category'],
                parse_dates=['Datetime_updated'],)
            .sort_values('Datetime_updated', ascending=False)
            .drop_duplicates(subset = ['Smart_contract', 'Buyer_address'], keep='first')
        )

        df_category = (
            df
            .loc[:, ['Category', 'Smart_contract', 'Buyer_address', ]]
        )

        # get owners collection distribution
        df_owners_pct = (
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
            df_category
            .merge(df_owners_pct, how = 'left', on = ['Buyer_address'])
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
        .loc[:, ['Art', 'Collectible', 'Games', 'Metaverse', 'Other', 'Utility']]
    )

    return (df_y, df_X)

def train_knn_classifier(y_train, X_train):
    from sklearn.neighbors import KNeighborsClassifier

    # Training the K-NN model on the Training set
    classifier = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    joblib.dump(classifier, folder/"knn.joblib")

def train_initial_knn_classifier() -> None:
    df_y, df_X = get_data_y_X()
    X = df_X.values
    y = df_y.iloc[:, 0].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    X_train = transform_X(X_train)
    train_knn_classifier(y_train, X_train)

def load_knn_classifier():
    path = folder/"knn.joblib"
    if not os.path.exists(path):
        train_initial_knn_classifier()
    classifier = joblib.load(path)
    return classifier

def transform_X(X_train):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    return X_train

def main():
    pass

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    # Predicting the Test set results
    df_y, df_X = get_data_y_X()
    X = df_X.values
    y = df_y.iloc[:, 0].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    # transform
    X_train = transform_X(X_train)
    X_test = transform_X(X_test)
    
    # train
    classifier = load_knn_classifier()
    y_pred = classifier.predict(X_test)
    labels = list(set(y_train))
    df_confusion = (
        pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels), 
        index = labels, columns = labels)
    )
    print(df_confusion)