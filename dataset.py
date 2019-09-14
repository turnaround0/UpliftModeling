import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


def load_data(dataset_name):
    # Link of hillstrom: https://drive.google.com/open?id=15osyN4c5z1pSo1JkxwL_N8bZTksRvQuU
    # Link of lalonde: https://drive.google.com/open?id=1b8N7WtwIe2WmQJD1KL5UAy70K13MxwKj
    # Link of criteo: https://drive.google.com/open?id=1Vxv7JiEyFr2A99xT6vYzB5ps5WhvV7NE
    if dataset_name == 'hillstrom':
        return pd.read_csv('Hillstrom.csv')
    elif dataset_name == 'lalonde':
        return pd.read_csv('Lalonde.csv')
    elif dataset_name == 'criteo':
        return pd.read_csv('criteo_small_fix.csv')
    else:
        return None


def save_json(name, data):
    with open(name + '.json', 'w') as f:
        json.dump(data, f)


def load_json(name):
    with open(name + '.json', 'r') as f:
        print('Open success')
        return json.load(f)


def create_fold(n_fold, seed, dataset_name, X, ty):
    # Create K-fold generator
    if dataset_name == 'hillstrom':
        fold_gen = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed).split(X, ty)
    elif dataset_name == 'lalonde':
        fold_gen = KFold(n_splits=n_fold, shuffle=True, random_state=seed).split(X)
    elif dataset_name == 'criteo':
        fold_gen = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed).split(X, ty)
    else:
        print('Invalid dataset name!')
        assert ()

    return fold_gen


def data_reindex(train_index, test_index, X, T, Y):
    X_train = X.reindex(train_index)
    X_test = X.reindex(test_index)
    Y_train = Y.reindex(train_index)
    Y_test = Y.reindex(test_index)
    T_train = T.reindex(train_index)
    T_test = T.reindex(test_index)

    return X_test, X_train, T_test, T_train, Y_test, Y_train
