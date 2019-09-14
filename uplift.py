#!/usr/bin/python3
import numpy as np
import pandas as pd
import time
import argparse
from sklearn.model_selection import StratifiedKFold, KFold

from config import wrapper_models, get_models, get_search_space
from dataset import load_data, save_json, load_json
from preprocess import preprocess_data, assign_data
from tune import do_general_wrapper_approach, do_tuning_parameters, do_niv_variable_selection, get_tuning_data_dict, \
    do_drop_vars
from measure import performance, qini
from plot import plot_all

# Parameters
seed = 1234
n_fold = 5
p_test = 0.33
n_niv_params = 50
enable_tune_parameters = True
repeat_num = 0  # For small dataset with regression


def print_overview(dataset_name, df, T, Y, ty):
    print('Shape:', df.shape)
    if dataset_name == 'lalonde':
        print('== Sum of each group ==')
        print(Y.groupby(T).sum())
        print('== Count of each group ==')
        print(T.groupby(T).count())
        print('== Average of each group ==')
        avg = Y.groupby(T).sum() / T.groupby(T).count()
        print(avg)
        print('Uplift:', avg[1] - avg[0])
    else:
        count = ty.groupby(ty).count()
        print('== Count of each group ==')
        print(count)
        uplift = count['TR'] / (count['TR'] + count['TN']) - count['CR'] / (count['CR'] + count['CN'])
        print('Uplift:', uplift)


def plot_data(dataset_names):
    for dataset_name in dataset_names:
        print('*** Dataset name:', dataset_name)
        qini_dict = load_json(dataset_name + '_qini')
        var_sel_dict = load_json(dataset_name + '_val_sel')
        plot_all(dataset_name, qini_dict, var_sel_dict)
    exit()


def data_reindex(train_index, test_index, X, T, Y):
    X_train = X.reindex(train_index)
    X_test = X.reindex(test_index)
    Y_train = Y.reindex(train_index)
    Y_test = Y.reindex(test_index)
    T_train = T.reindex(train_index)
    T_test = T.reindex(test_index)

    return X_test, X_train, T_test, T_train, Y_test, Y_train


def create_fold(dataset_name, X, ty):
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


def main():
    parser = argparse.ArgumentParser(description='***** Uplift modeling *****')
    parser.add_argument('-d', action='store_true', help='Only loading json files and display plots')
    args = parser.parse_args()

    dataset_names = ['hillstrom', 'lalonde', 'criteo']

    # Display only plots and tables with -d option
    if args.d:
        plot_data(dataset_names)

    for dataset_name in dataset_names:
        print('*** Dataset name:', dataset_name)
        start_time = time.time()

        # Load data with preprocessing
        df = load_data(dataset_name)
        df = preprocess_data(df, dataset=dataset_name)
        if repeat_num:
            df = pd.concat([df] * repeat_num, axis=0).reset_index(drop=True)
        X, Y, T, ty = assign_data(df)

        print_overview(dataset_name, df, T, Y, ty)

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

            fold_gen = create_fold(dataset_name, X, ty)

            for idx, (train_index, test_index) in enumerate(fold_gen):
                print('Fold #{}'.format(idx + 1))
                fold_start_time = time.time()

                X_test, X_train, T_test, T_train, Y_test, Y_train = data_reindex(train_index, test_index, X, T, Y)

                niv_done = False
                if X_train.shape[1] > n_niv_params:
                    niv_done = True
                    X_test, X_train = do_niv_variable_selection(X_test, X_train, T_train, Y_train, n_niv_params)

                data_dict = get_tuning_data_dict(X_train, Y_train, T_train, dataset_name, p_test, seed)

                if enable_wrapper and not niv_done:
                    qini_values = do_general_wrapper_approach(models[model_name], search_space,
                                                              data_dict, X_test, X_train)
                    var_sel_dict[model_name].append(qini_values)

                if enable_tune_parameters:
                    best_params = do_tuning_parameters(model_name, models[model_name], data_dict, search_space)
                else:
                    best_params = {}

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
