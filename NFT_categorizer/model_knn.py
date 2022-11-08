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

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import os
import joblib 
# %%
folder = cfg.dir_nft_categorizer / "data"

def transform_X(X_train):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    return X_train

def train_knn_classifier(y_train, X_train):
    from sklearn.neighbors import KNeighborsClassifier
    X_train = transform_X(X_train)

    # Training the K-NN model on the Training set
    classifier = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    return classifier

def knn_predict(classifier, X_train):
    X_train = transform_X(X_train)
    y_pred = classifier.predict(X_train)
    return y_pred

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

def main():
   pass

if __name__ == '__main__':
    main()