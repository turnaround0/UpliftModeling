#!/usr/bin/python3
import numpy as np
import pandas as pd
import time
import json
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from config import urf_methods, wrapper_models, get_models, get_search_space
from preprocess import preprocess_data, assign_data
from tune import parameter_tuning, wrapper, do_niv_variable_selection
from experiment import performance, qini

from plot import plot_fig5, plot_table6, plot_fig7, plot_fig8, plot_fig9


def insert_urf_method(model_name):
    if model_name in urf_methods.keys():
        return {'method': urf_methods[model_name]}
    else:
        return {}


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


def display_results(dataset_name, qini_dict, var_sel_dict):
    if dataset_name != 'criteo':
        plot_fig5(dataset_name, var_sel_dict)
    plot_table6(dataset_name, qini_dict)
    plot_fig7(dataset_name, qini_dict)
    plot_fig8(dataset_name, qini_dict)
    plot_fig9(dataset_name, qini_dict)


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


def do_general_wrapper_approach(model, data_dict, search_space):
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

    wrapper_end_time = time.time()
    print('Wrapper time:', wrapper_end_time - wrapper_start_time)

    return best_drop_vars, qini_values


def do_tuning_parameters(model, data_dict, search_space):
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

    return best_params


def main():
    parser = argparse.ArgumentParser(description='***** Uplift modeling *****')
    parser.add_argument('-d', action='store_true', help='Only loading json files and display plots')
    args = parser.parse_args()

    dataset_names = ['hillstrom', 'lalonde', 'criteo']

    # Display only plots and tables with -d option
    if args.d:
        for dataset_name in dataset_names:
            print('*** Dataset name:', dataset_name)
            qini_dict = load_json(dataset_name + '_qini')
            var_sel_dict = load_json(dataset_name + '_val_sel')
            display_results(dataset_name, qini_dict, var_sel_dict)
        exit()

    # Parameters
    seed = 1234
    n_fold = 5
    p_test = 0.33
    n_niv_params = 50
    enable_tune_parameters = True
    repeat_num = 0  # For small dataset with regression

    for dataset_name in dataset_names:
        print('*** Dataset name:', dataset_name)
        start_time = time.time()

        # Load data with preprocessing
        df = load_data(dataset_name)
        df = preprocess_data(df, dataset=dataset_name)
        if repeat_num:
            df = pd.concat([df] * repeat_num, axis=0).reset_index(drop=True)
        X, Y, T, ty = assign_data(df)

        print('Shape:', df.shape)

        if dataset_name == 'lalonde':
            print('== Sum of each group ==')
            print(Y.groupby(T).sum())
            print('== Count of each group ==')
            print(T.groupby(T).count())
            print('== Average of each group ==')
            avg = Y.groupby(T).sum() / T.groupby(T).count()
            print('Uplift:', avg[1] - avg[0])
        else:
            count = ty.groupby(ty).count()
            print('== Count of each group ==')
            print(count)
            uplift = count['TR'] / (count['TR'] + count['TN']) - count['CR'] / (count['CR'] + count['CN'])
            print('Uplift:', uplift)

        # Cross-validation with K-fold
        qini_dict = {}
        var_sel_dict = {}
        models = get_models(dataset_name)
        for model_name in models:
            print('* Model:', model_name)

            var_sel_dict[model_name] = []
            qini_dict[model_name] = []
            qini_list = []
            enable_wrapper = model_name in wrapper_models
            search_space = get_search_space(dataset_name, model_name)

            fit = models[model_name].fit
            predict = models[model_name].predict

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

            for idx, (train_index, test_index) in enumerate(fold_gen):
                print('Fold #{}'.format(idx + 1))
                fold_start_time = time.time()

                X_train = X.reindex(train_index)
                X_test = X.reindex(test_index)
                Y_train = Y.reindex(train_index)
                Y_test = Y.reindex(test_index)
                T_train = T.reindex(train_index)
                T_test = T.reindex(test_index)

                niv_done = False
                if X_train.shape[1] > n_niv_params:
                    niv_start_time = time.time()
                    print('Start NIV variable selection')

                    survived_vars = do_niv_variable_selection(X_train, Y_train, T_train, n_niv_params)
                    print('NIV:', list(survived_vars))

                    X_train = X_train[survived_vars]
                    X_test = X_test[survived_vars]
                    niv_done = True

                    niv_end_time = time.time()
                    print('NIV time:', niv_end_time - niv_start_time)

                data_dict = get_tuning_data_dict(X_train, Y_train, T_train, dataset_name, p_test, seed)

                if enable_wrapper and not niv_done:
                    best_drop_vars, qini_values = do_general_wrapper_approach(models[model_name],
                                                                              data_dict, search_space)
                    print('Drop vars:', best_drop_vars)
                    var_sel_dict[model_name].append(qini_values)

                    data_dict['x_train'].drop(best_drop_vars, axis=1, inplace=True)
                    data_dict['x_test'].drop(best_drop_vars, axis=1, inplace=True)
                    X_train.drop(best_drop_vars, axis=1, inplace=True)
                    X_test.drop(best_drop_vars, axis=1, inplace=True)

                if enable_tune_parameters:
                    best_params = do_tuning_parameters(models[model_name], data_dict, search_space)
                    print('Best params:', best_params)
                else:
                    best_params = {}

                # In case of Uplift Random Forest tree,
                # split criterion (ed, kl or others) should be set.
                best_params.update(insert_urf_method(model_name))

                # Train model and predict outcomes
                mdl = fit(X_train, Y_train, T_train, **best_params)
                pred = predict(mdl, X_test, t=T_test)

                # Perform to check performance with Qini curve
                perf = performance(pred['pr_y1_t1'], pred['pr_y1_t0'], Y_test, T_test)
                q = qini(perf, plotit=False)

                # Store Qini values
                print('Qini =', q['qini'])
                qini_dict[model_name].append(q)
                qini_list.append(q['qini'])

                fold_end_time = time.time()
                print('Fold time:', fold_end_time - fold_start_time)

            print('Qini values: ', qini_list)
            print('    mean: {}, std: {}'.format(np.mean(qini_list), np.std(qini_list)))

        end_time = time.time()
        print('Total time:', end_time - start_time)

        save_json(dataset_name + '_val_sel', var_sel_dict)
        save_json(dataset_name + '_qini', qini_dict)

        # display_results(dataset_name, qini_dict, var_sel_dict)


if __name__ == '__main__':
    main()
