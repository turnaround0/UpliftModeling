import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tune.niv import niv_variable_selection
from tune.param import parameter_tuning
from tune.wrapper import wrapper
from utils.utils import load_json, save_json


def get_tuning_data_dict(X_train, Y_train, T_train, dataset_name, p_test, seed):
    df = X_train.copy()
    df['Y'] = Y_train
    df['T'] = T_train

    if dataset_name in ['hillstrom', 'criteo']:
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

    return {
        "x_train": X_tuning,
        "y_train": Y_tuning,
        "t_train": T_tuning,
        "x_test": X_validate,
        "y_test": Y_validate,
        "t_test": T_validate,
    }


def do_drop_vars(best_drop_vars, data_dict, X_test, X_train):
    data_dict['x_train'].drop(best_drop_vars, axis=1, inplace=True)
    data_dict['x_test'].drop(best_drop_vars, axis=1, inplace=True)
    X_train.drop(best_drop_vars, axis=1, inplace=True)
    X_test.drop(best_drop_vars, axis=1, inplace=True)


def do_general_wrapper_approach(model, search_space, data_dict, X_test, X_train):
    wrapper_start_time = time.time()
    print('Start wrapper variable selection')

    model_method = search_space.get('method', None)
    params = {'method': None if model_method is None else model_method[0]}
    if params['method'] == LogisticRegression:
        solver = search_space.get('solver', None)
        params['solver'] = None if solver is None else solver[0]

    _, drop_vars, qini_values = wrapper(model.fit, model.predict, data_dict, params=params)
    best_qini = max(qini_values)
    best_idx = qini_values.index(best_qini)
    best_drop_vars = drop_vars[:best_idx]
    print('Drop vars:', best_drop_vars)

    wrapper_end_time = time.time()
    print('Wrapper time:', wrapper_end_time - wrapper_start_time)

    do_drop_vars(best_drop_vars, data_dict, X_test, X_train)

    return qini_values


def do_tuning_parameters(model, search_space, data_dict):
    keys = search_space.keys()
    n_space = np.prod([len(search_space[key]) for key in keys])

    # If number of space is 1, we don't need to perform tuning parameters
    if n_space > 1:
        tune_start_time = time.time()
        print('Start parameter tuning')

        _, best_params = parameter_tuning(model.fit, model.predict, data_dict,
                                          search_space=search_space)

        tune_end_time = time.time()
        print('Tune time:', tune_end_time - tune_start_time)
    else:
        best_params = {key: search_space[key][0] for key in keys}

    print('Best params:', best_params)

    return best_params


def do_niv(X_test, X_train, T_train, Y_train, n_niv_params, dataset_name, fold_idx):
    niv_filename = 'niv_' + dataset_name
    fold_name = 'fold' + str(fold_idx + 1)
    niv_vars = load_json(niv_filename)
    survived_vars = niv_vars.get(fold_name) if niv_vars else None

    if survived_vars:
        print('Stored NIV:', survived_vars)
        X_test = X_test[survived_vars]
        X_train = X_train[survived_vars]
    else:
        niv_start_time = time.time()
        print('Start NIV variable selection')

        survived_vars = niv_variable_selection(X_train, Y_train, T_train, n_niv_params)
        print('NIV:', list(survived_vars))

        X_train = X_train[survived_vars]
        X_test = X_test[survived_vars]

        niv_end_time = time.time()
        print('NIV time:', niv_end_time - niv_start_time)

        if niv_vars:
            niv_vars.update({fold_name: survived_vars.tolist()})
        else:
            niv_vars = {fold_name: survived_vars.tolist()}
        save_json(niv_filename, niv_vars)

    return X_test, X_train

