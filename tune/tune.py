import time
import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from experiment.measure import performance, qini
from tune.niv import niv_variable_selection
from tune.param import parameter_tuning
from tune.wrapper import wrapper
from utils.utils import num_class

calc_mlai = True


def find_mlai_best_params(fit_mdl, pred_mdl, data, model_params, mlai_pairs):
    x_train = data['x_train']
    y_train = data['y_train']
    t_train = data['t_train']
    x_test = data['x_test']
    y_test = data['y_test']
    t_test = data['t_test']

    max_q = -float('inf')
    best_pair = None

    mdl = fit_mdl(x_train, y_train, t_train, **model_params)

    for i, pair in enumerate(mlai_pairs):
        pred = pred_mdl(mdl, newdata=x_test, y=y_test, t=t_test, alpha=pair[0], beta=pair[1])

        try:
            perf = performance(pred['pr_y1_t1'], pred['pr_y1_t0'], y_test, t_test)
        except Exception as e:
            print(e)
            continue

        q = qini(perf, plotit=False)['qini']
        if q > max_q:
            max_q = q
            best_pair = pair

    return best_pair


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


def do_niv_variable_selection(X_test, X_train, T_train, Y_train, n_niv_params):
    niv_start_time = time.time()
    print('Start NIV variable selection')

    survived_vars = niv_variable_selection(X_train, Y_train, T_train, n_niv_params)
    print('NIV:', list(survived_vars))

    X_train = X_train[survived_vars]
    X_test = X_test[survived_vars]

    niv_end_time = time.time()
    print('NIV time:', niv_end_time - niv_start_time)

    return survived_vars, X_test, X_train


def do_find_best_mlai_params(model, model_params, mlai_params, data_dict):
    tune_start_time = time.time()
    print('Start finding best MLAI params')

    if calc_mlai:
        df = data_dict['x_train'].copy()
        df['Y'] = data_dict['y_train']
        df['T'] = data_dict['t_train']

        tr, tn, cr, cn = num_class(df, 'Y', 'T')
        pr_y1_t1 = tr / (tr + tn)
        pr_y1_t0 = cr / (cr + cn)
        pr_y0_t1 = tn / (tr + tn)
        pr_y0_t0 = cn / (cr + cn)

        # similarity for cn with tr
        ed_gain = (pr_y1_t1 - pr_y0_t0) ** 2
        gini_tr_cn = 2 * pr_y1_t1 * pr_y0_t0 * (1 - pr_y1_t1) * (1 - pr_y0_t0)
        gini_tr = 2 * pr_y1_t1 * (1 - pr_y1_t1)
        gini_cn = 2 * pr_y0_t0 * (1 - pr_y0_t0)
        ed_norm = gini_tr_cn * ed_gain + gini_tr * pr_y1_t1 + gini_cn * pr_y0_t0 + 0.5
        tr_cn = ed_gain / ed_norm

        # similarity for tn with cr
        ed_gain = (pr_y0_t1 - pr_y1_t0) ** 2
        gini_tn_cr = 2 * pr_y0_t1 * pr_y1_t0 * (1 - pr_y0_t1) * (1 - pr_y1_t0)
        gini_tn = 2 * pr_y0_t1 * (1 - pr_y0_t1)
        gini_cr = 2 * pr_y1_t0 * (1 - pr_y1_t0)
        ed_norm = gini_tn_cr * ed_gain + gini_tn * pr_y0_t1 + gini_cr * pr_y1_t0 + 0.5
        cr_tn = ed_gain / ed_norm

        print('Best MLAI params:', tr_cn, cr_tn)
        return tr_cn, cr_tn
    else:
        mlai_pairs = list(itertools.product(mlai_params, mlai_params))

    best_params = find_mlai_best_params(model.fit, model.predict, data_dict, model_params, mlai_pairs)

    tune_end_time = time.time()
    print('Tune time:', tune_end_time - tune_start_time)
    print('Best MLAI params:', best_params[0], best_params[1])

    return best_params
