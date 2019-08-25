#!/usr/bin/python3
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from preprocess import preprocess_data, assign_data
from tune import parameter_tuning, wrapper
from experiment import performance, qini
from models import model_tma

# Hyper-parameters
search_space = {
    'method': [LogisticRegression],
    'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
    'penalty': ['none', 'l2'],
    'tol': [1e-2, 1e-3, 1e-4],
    'C': [1e6, 1e3, 1, 1e-3, 1e-6],
}


# Search space for tree
# search_space = {
#     'ntree': [10, ],
#     'mtry': [3, ],
#     'bagging_fraction': [0.6, ],
#     'method': ['ED',],
#     'max_depth': [10, ],
#     'min_split': [1000, ],
#     'min_bucket_t0': [100,],
#     'min_bucket_t1': [100,],
# }

# Search space for dta
# search_space = {
#     'solver': ['liblinear', ],
# }

models = {
    'tma': model_tma,
    # 'dta': model_dta,
    # 'lai': model_lai,
    # 'glai': model_glai,
    # 'rvtu': model_rvtu,
    # 'tree': model_tree,
    # 'urf': model_rf,
}


def load_data(dataset_name):
    # Link of hillstrom: https://drive.google.com/open?id=15osyN4c5z1pSo1JkxwL_N8bZTksRvQuU
    # Link of lalonde: https://drive.google.com/open?id=1b8N7WtwIe2WmQJD1KL5UAy70K13MxwKj
    # Link of criteo: https://drive.google.com/open?id=1Vxv7JiEyFr2A99xT6vYzB5ps5WhvV7NE
    if dataset_name == 'hillstrom':
        return pd.read_csv('Hillstrom.csv')
    elif dataset_name == 'lalonde':
        return pd.read_csv('Lalonde.csv')
    elif dataset_name == 'criteo':
        return pd.read_csv('criteo_small.csv')
    else:
        return None


def main():
    # Parameters
    dataset_name = 'hillstrom'
    seed = 1234
    n_fold = 5
    p_test = 0.33
    enable_tune_parameters = False
    enable_wrapper = False

    # Load data with preprocessing
    df = load_data(dataset_name)
    df = preprocess_data(df)
    X, Y, T, ty = assign_data(df)

    # Create K-fold generator
    if dataset_name == 'hillstrom':
        fold_gen = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed).split(X, ty)
    elif dataset_name == 'lalonde':
        fold_gen = KFold(n_splits=n_fold, shuffle=True, random_state=seed).split(X)
    elif dataset_name == 'criteo':
        fold_gen = KFold(n_splits=n_fold, shuffle=True, random_state=seed).split(X)
    else:
        print('Invalid dataset name!')
        assert()

    # Cross-validation with K-fold
    qini_list = []
    for model_name in models:
        fit = models[model_name].fit
        predict = models[model_name].predict

        for idx, (train_index, test_index) in enumerate(fold_gen):
            X_train = X.reindex(train_index)
            X_test = X.reindex(test_index)
            Y_train = Y.reindex(train_index)
            Y_test = Y.reindex(test_index)
            T_train = T.reindex(train_index)
            T_test = T.reindex(test_index)

            if enable_wrapper or enable_tune_parameters:
                df = X_train.copy()
                df['Y'] = Y_train
                df['T'] = T_train

                if dataset_name == 'hillstrom':
                    stratify = df[['Y', 'T']]
                else:
                    stratify = T_train
                tuning_df, validate_df = train_test_split(
                    df, test_size=p_test, random_state=seed, stratify=stratify)

                X_tuning = tuning_df.drop(['Y', 'T'], axis=1)
                Y_tuning = tuning_df['Y']
                T_tuning = tuning_df['T']

                X_validate = validate_df.drop(['Y', 'T'], axis=1)
                Y_validate = validate_df['Y']
                T_validate = validate_df['T']

                data_dict = {
                    "x_train": X_tuning,
                    "y_train": Y_tuning,
                    "t_train": T_tuning,
                    "x_test": X_validate,
                    "y_test": Y_validate,
                    "t_test": T_validate,
                }

                if enable_wrapper:
                    model_method = search_space.get('method', None)
                    params = {
                        'method': None if model_method is None else model_method[0],
                    }
                    if params['method'] == LogisticRegression:
                        solver = search_space.get('solver', None)
                        params['solver'] = None if solver is None else solver[0]

                    _, drop_vars, qini_values = wrapper(fit, predict, data_dict, params=params)
                    best_qini = max(qini_values)
                    best_idx = qini_values.index(best_qini)
                    best_drop_vars = drop_vars[:best_idx]

                    X_tuning.drop(best_drop_vars, axis=1, inplace=True)
                    X_validate.drop(best_drop_vars, axis=1, inplace=True)
                    X_train.drop(best_drop_vars, axis=1, inplace=True)
                    X_test.drop(best_drop_vars, axis=1, inplace=True)

                if enable_tune_parameters:
                    _, best_params = parameter_tuning(fit, predict, data_dict,
                                                      search_space=search_space)

                    """
                    best_params = {k: v[0] for k, v in search_space.items()}
                    q = qini(perf)
                    q_list.append(q)
                    print("Best_params: ", best_params)
                    """

            # Train model and predict outcomes
            best_params = {}
            mdl = fit(X_train, Y_train, T_train, **best_params)
            pred = predict(mdl, X_test)

            # Perform to check performance with Qini curve
            perf = performance(pred['pr_y1_t1'], pred['pr_y1_t0'], Y_test, T_test)
            q = qini(perf, plotit=False)
            qini_list.append(q['qini'])

    print('Qini values: ', qini_list)
    print('    mean: {}, std: {}'.format(np.mean(qini_list), np.std(qini_list)))

    """
    print("Method: {}".format(method))
    print("search space:", search_space)
    qini_list = [q['qini'] for q in q_list]
    """


if __name__ == '__main__':
    main()
