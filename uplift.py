#!/usr/bin/python3
import numpy as np
import time
import argparse

from config.config import ConfigSet
from dataset.dataset import load_data, create_fold, data_reindex
from dataset.preprocess import preprocess_data, assign_data
from utils.utils import save_json
from utils.helper import print_overview, plot_data
from tune.tune import do_general_wrapper_approach, do_tuning_parameters, get_tuning_data_dict, do_niv
from experiment.measure import performance, qini
from experiment.plot import plot_table6

# Parameters
seed = 1234
n_fold = 5
p_test = 0.33
n_niv_params = 50


def set_model_specific_params(config_set, dataset_name, model, model_name):
    if model_name.endswith('_ext'):
        max_round = config_set.get_option('max_round', dataset_name, model_name)
        p_value = config_set.get_option('p_value', dataset_name, model_name)
        model.set_params(max_round, p_value)
    elif model_name.endswith('_focus'):
        p_value = config_set.get_option('p_value', dataset_name, model_name)
        model.set_params(p_value)
    elif 'mlai' in model_name:
        class_weight = config_set.get_option('class_weight', dataset_name, model_name)
        model.set_params(class_weight)


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

                set_model_specific_params(config_set, dataset_name, model, model_name)

                if config_set.is_enable('niv') and X_train.shape[1] > n_niv_params:
                    X_test, X_train = do_niv(X_test, X_train, T_train, Y_train, n_niv_params, dataset_name, idx)

                over_sampling = config_set.get_option('over_sampling', dataset_name, model_name)
                if over_sampling:
                    params_over = config_set.get_option('params_over', dataset_name, model_name)
                    X_train, T_train, Y_train = over_sampling(X_train, T_train, Y_train, params_over)

                data_dict = get_tuning_data_dict(X_train, Y_train, T_train, dataset_name, p_test, seed)
                search_space = config_set.get_search_space(dataset_name, model_name)

                if config_set.is_enable('wrapper', dataset_name, model_name) and X_train.shape[1] <= n_niv_params:
                    qini_values = do_general_wrapper_approach(model, search_space, data_dict, X_test, X_train)
                    var_sel_dict[model_name].append(qini_values)

                if config_set.is_enable('tune'):
                    best_params = do_tuning_parameters(model, search_space, data_dict)
                else:
                    best_params = config_set.get_default_params(dataset_name, model_name)
                print('Params:', best_params)

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

        save_json(dataset_name + '_val_sel', var_sel_dict)
        save_json(dataset_name + '_qini', qini_dict)

        end_time = time.time()
        print('Total time:', end_time - start_time)

        plot_table6(dataset_name, qini_dict)


if __name__ == '__main__':
    main()
