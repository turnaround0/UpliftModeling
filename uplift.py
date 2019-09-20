#!/usr/bin/python3
import numpy as np
import time
import argparse

from config.config import ConfigSet
from dataset import load_data, save_json, create_fold, data_reindex
from helper import print_overview, plot_data, get_uplift
from preprocess import preprocess_data, assign_data
from tune import do_general_wrapper_approach, do_tuning_parameters, do_niv_variable_selection,\
    get_tuning_data_dict, do_find_best_mlai_params
from measure import performance, qini
from plot import plot_table6

# Parameters
seed = 1234
n_fold = 5
p_test = 0.33
n_niv_params = 50


def set_model_specific_params(config_set, dataset_name, model, model_name, T_train, Y_train):
    if model_name.endswith('_ext'):
        max_round = config_set.get_option('max_round', dataset_name, model_name)
        u_list = config_set.get_option('u_list', dataset_name, model_name)
        train_uplift = get_uplift(dataset_name, T_train, Y_train)
        print('Train uplift:', train_uplift)
        model.set_params(max_round, u_list, train_uplift)
    elif model_name.endswith('_focus'):
        u_value = config_set.get_option('u_value', dataset_name, model_name)
        train_uplift = get_uplift(dataset_name, T_train, Y_train)
        print('Train uplift:', train_uplift)
        model.set_params(u_value, train_uplift)
    elif model_name.endswith('_ext2'):
        max_round = config_set.get_option('max_round', dataset_name, model_name)
        p_list = config_set.get_option('p_list', dataset_name, model_name)
        model.set_params(max_round, p_list)
    elif model_name.endswith('_focus2'):
        p_value = config_set.get_option('p_value', dataset_name, model_name)
        model.set_params(p_value)


def main():
    parser = argparse.ArgumentParser(description='***** Uplift modeling *****')
    parser.add_argument('-d', action='store_true', help='Draw: Only loading json files and display plots')
    parser.add_argument('-s', action='store', help='Set: Config set name (ex: -s test_all')
    args = parser.parse_args()

    if args.s:
        config_set = ConfigSet(args.s)
    else:
        config_set = ConfigSet('test_all')
    dataset_names = config_set.get_dataset_names()

    # Display only plots and tables with -d option
    if args.d:
        plot_data(dataset_names)
        exit()

    for dataset_name in dataset_names:
        print('*** Dataset name:', dataset_name)
        start_time = time.time()

        qini_dict = {}
        var_sel_dict = {}

        # Load data with preprocessing
        df = load_data(dataset_name)
        df = preprocess_data(df, dataset=dataset_name)
        X, Y, T, ty = assign_data(df)

        pre_time = time.time()
        print('Load and pre-processing time:', pre_time - start_time)

        print_overview(dataset_name, df, T, Y, ty)

        model_names = config_set.get_model_names(dataset_name)
        niv_results = []

        for model_name in model_names:
            print('* Model:', model_name)

            var_sel_dict[model_name] = []
            qini_dict[model_name] = []
            qini_list = []

            model = config_set.get_model(dataset_name, model_name)
            fit = model.fit
            predict = model.predict

            fold_gen = create_fold(n_fold, seed, dataset_name, X, ty)

            for idx, (train_index, test_index) in enumerate(fold_gen):
                print('Fold #{}'.format(idx + 1))
                fold_start_time = time.time()

                X_test, X_train, T_test, T_train, Y_test, Y_train = data_reindex(train_index, test_index, X, T, Y)

                set_model_specific_params(config_set, dataset_name, model, model_name, T_train, Y_train)

                if config_set.is_enable('niv') and X_train.shape[1] > n_niv_params:
                    if idx >= len(niv_results):
                        survived_vars, X_test, X_train = \
                            do_niv_variable_selection(X_test, X_train, T_train, Y_train, n_niv_params)
                        niv_results.append(survived_vars)
                    else:
                        survived_vars = niv_results[idx]
                        X_test = X_test[survived_vars]
                        X_train = X_train[survived_vars]

                over_sampling = config_set.get_option('over_sampling', dataset_name, model_name)
                if over_sampling:
                    X_train, T_train, Y_train = over_sampling(X_train, T_train, Y_train)

                data_dict = get_tuning_data_dict(X_train, Y_train, T_train, dataset_name, p_test, seed)
                search_space = config_set.get_search_space(dataset_name, model_name)

                if config_set.is_enable('wrapper', dataset_name, model_name) and X_train.shape[1] <= n_niv_params:
                    qini_values = do_general_wrapper_approach(model, search_space, data_dict, X_test, X_train)
                    var_sel_dict[model_name].append(qini_values)

                if config_set.is_enable('tune'):
                    best_params = do_tuning_parameters(model, search_space, data_dict)
                else:
                    best_params = config_set.get_default_params(dataset_name, model_name)

                mlai_params = config_set.get_option('mlai_values', dataset_name, model_name)
                # if mlai_params:
                #     pred = predict(mdl, X_test, t=T_test, alpha=alpha, beta=beta)
                # else:
                #     # Train model and predict outcomes
                class_weight = config_set.get_option('class_weight', dataset_name, model_name)
                print(class_weight)
                if class_weight == 'calculate':
                    alpha, beta = do_find_best_mlai_params(model, best_params, mlai_params, data_dict)
                    best_params.update({'class_weight': {'CN': alpha, 'CR': 1, 'TN': beta, 'TR': 1}})
                else:
                    best_params.update({'class_weight': class_weight})

                print('Params:', best_params)
                # Train model and predict outcomes
                mdl = fit(X_train, Y_train, T_train, **best_params)

                # mlai_params = config_set.get_option('mlai_values', dataset_name, model_name)
                # if mlai_params:
                #     alpha, beta = do_find_best_mlai_params(model, best_params, mlai_params, data_dict)
                #     pred = predict(mdl, X_test, t=T_test, alpha=alpha, beta=beta)
                # else:
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

        save_json(dataset_name + '_val_sel', var_sel_dict)
        save_json(dataset_name + '_qini', qini_dict)

        end_time = time.time()
        print('Total time:', end_time - start_time)

        plot_table6(dataset_name, qini_dict)


if __name__ == '__main__':
    main()
