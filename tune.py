import time
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from measure import performance, qini


def parameter_tuning(fit_mdl, pred_mdl, data, search_space, plotit=False):
    """
    Given a model, search all combination of parameter sets and find
    the best parameter set

    Args:
        fit_mdl: model function
        pred_mdl: predict function of fit_mdl
        data:
            {
                "x_train": predictor variables of training dataset,
                "y_train": target variables of training dataset,
                "t_train": treatment variables of training dataset,
                "x_test": predictor variables of test (usually, validation) dataset,
                "y_test": target variables of test (usually, validation) dataset,
                "t_test": treatment variables of test (usually, validation) dataset,
            }
        search_space:
            {
                parameter_name: [search values]
            }
        plotit: draw plot if True, otherwise don't draw it
    Return:
        The best parameter set
    """
    x_train = data['x_train']
    y_train = data['y_train']
    t_train = data['t_train']
    x_test = data['x_test']
    y_test = data['y_test']
    t_test = data['t_test']

    max_q = -float('inf')
    best_mdl = None

    keys = search_space.keys()
    n_space = [len(search_space[key]) for key in keys]
    n_iter = int(np.prod(n_space))

    best_params = None
    for i in range(n_iter):
        params = {}
        for idx, key in enumerate(keys):
            params[key] = search_space[key][i % n_space[idx]]
            i = int(i / n_space[idx])

        mdl = fit_mdl(x_train, y_train, t_train, **params)
        pred = pred_mdl(mdl, newdata=x_test, y=y_test, t=t_test)

        try:
            perf = performance(pred['pr_y1_t1'], pred['pr_y1_t0'], y_test, t_test)
        except Exception as e:
            print(e)
            continue
        q = qini(perf, plotit=plotit)['qini']
        if plotit:
            print(q, params)
        if q > max_q:
            max_q = q
            best_mdl = mdl
            best_params = params

    return best_mdl, best_params


def wrapper(fit_mdl, pred_mdl, data, params=None,
            best_models=None, drop_variables=None, qini_values=None):
    """
    General wrapper approach

    Args:
        fit_mdl: model function
        pred_mdl: predict function of fit_mdl
        data:
            {
                "x_train": predictor variables of training dataset,
                "y_train": target variables of training dataset,
                "t_train": treatment variables of training dataset,
                "x_test": predictor variables of test (usually, validation) dataset,
                "y_test": target variables of test (usually, validation) dataset,
                "t_test": treatment variables of test (usually, validation) dataset,
            }
    Return:
        (A list of best models, The list of dropped variables)
    """
    if best_models is None:
        best_models = []
    if drop_variables is None:
        drop_variables = []
    if qini_values is None:
        qini_values = []
    if params is None:
        params = {}

    x_train = data['x_train']
    y_train = data['y_train']
    t_train = data['t_train']
    x_test = data['x_test']
    y_test = data['y_test']
    t_test = data['t_test']

    variables = data['x_train'].columns

    max_q = -float('inf')
    drop_var = None
    best_mdl = None
    for var in variables:
        if var in drop_variables:
            continue
        x = x_train.copy()
        x.drop(drop_variables + [var], axis=1, inplace=True)
        mdl = fit_mdl(x, y_train, t_train, **params)
        x = x_test.copy()
        x.drop(drop_variables + [var], axis=1, inplace=True)
        pred = pred_mdl(mdl, newdata=x, y=y_test, t=t_test)
        perf = performance(pred['pr_y1_t1'], pred['pr_y1_t0'], y_test, t_test)
        q = qini(perf, plotit=False)['qini']
        if q > max_q:
            max_q = q
            drop_var = var
            best_mdl = mdl

    best_models.append(best_mdl)
    drop_variables.append(drop_var)
    qini_values.append(max_q)

    left_vars = [var for var in variables if (var not in drop_variables)]

    if len(variables) == len(drop_variables) + 1:
        return best_models, drop_variables + left_vars, qini_values
    else:
        return wrapper(fit_mdl, pred_mdl, data, params=params,
                       best_models=best_models, drop_variables=drop_variables,
                       qini_values=qini_values)


def niv_variable_selection(x, y, t, max_vars):
    """
    NIV variable selection procedure

    Args:
        x: predictor variables of training dataset,
        y: target variables of training dataset,
        t: treatment variables of training dataset,
        max_vars: maximum number of return variables,
    Return:
        (The list of survived variables)
    """
    y1_t = (y == 1) & (t == 1)
    y0_t = (y == 0) & (t == 1)
    y1_c = (y == 1) & (t == 0)
    y0_c = (y == 0) & (t == 0)

    sum_y1_t = sum(y1_t)
    sum_y0_t = sum(y0_t)
    sum_y1_c = sum(y1_c)
    sum_y0_c = sum(y0_c)

    niv_dict = {}
    for col in x.columns:
        df = pd.concat([x[col].rename(col), y1_t.rename('y1_t'), y0_t.rename('y0_t'),
                        y1_c.rename('y1_c'), y0_c.rename('y0_c')], axis=1)
        x_group = df.groupby(x[col])
        x_sum = x_group.sum()

        if sum_y0_t == 0 or sum_y1_t == 0:
            woe_t = 0
        else:
            woe_t = x_sum.apply(lambda r: np.log((r['y1_t'] * sum_y0_t) / (r['y0_t'] * sum_y1_t))
                                if r['y1_t'] > 0 and r['y0_t'] > 0 else 0, axis=1)

        if sum_y0_c == 0 or sum_y1_c == 0:
            woe_c = 0
        else:
            woe_c = x_sum.apply(lambda r: np.log((r['y1_c'] * sum_y0_c) / (r['y0_c'] * sum_y1_c))
                                if r['y1_c'] > 0 and r['y0_c'] > 0 else 0, axis=1)

        nwoe = woe_t - woe_c

        p_x_y1_t = x_sum['y1_t'] / sum_y1_t if sum_y1_t > 0 else 0
        p_x_y0_t = x_sum['y0_t'] / sum_y0_t if sum_y0_t > 0 else 0
        p_x_y1_c = x_sum['y1_c'] / sum_y1_c if sum_y1_c > 0 else 0
        p_x_y0_c = x_sum['y0_c'] / sum_y0_c if sum_y0_c > 0 else 0
        niv_weight = (p_x_y1_t * p_x_y0_c - p_x_y0_t * p_x_y1_c)

        niv_row = 100 * nwoe * niv_weight
        niv = niv_row.sum()
        niv_dict[col] = niv

    s_niv = pd.Series(niv_dict)
    s_selected_niv = s_niv.sort_values(ascending=False)[: max_vars]

    return s_selected_niv.index


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

    return X_test, X_train


def do_find_best_mlai_params(model, model_params, mlai_params, data_dict):
    tune_start_time = time.time()
    print('Start finding best MLAI params')

    """
    df = data_dict['x_train'].copy()
    df['Y'] = data_dict['y_train']
    df['T'] = data_dict['t_train']

    tr, tn, cr, cn = num_class(df, 'Y', 'T')
    pr_y1_t1 = tr / (tr + tn)
    pr_y1_t0 = cr / (cr + cn)
    pr_y0_t1 = tn / (tr + tn)
    pr_y0_t0 = cn / (cr + cn)

    tr_cn = [np.sum(pr_y1_t1 * np.log(pr_y1_t1 / pr_y0_t0)),
             np.sum(pr_y0_t0 * np.log(pr_y0_t0) / pr_y1_t1)]
    cr_tn = [np.sum(pr_y1_t0 * np.log(pr_y1_t0 / pr_y0_t1)),
             np.sum(pr_y0_t1 * np.log(pr_y0_t1 / pr_y1_t0))]

    mlai_pairs = list(itertools.product(tr_cn, cr_tn))
    """
    
    mlai_pairs = list(itertools.product(mlai_params, mlai_params))

    best_params = find_mlai_best_params(model.fit, model.predict, data_dict, model_params, mlai_pairs)

    tune_end_time = time.time()
    print('Tune time:', tune_end_time - tune_start_time)
    print('Best MLAI params:', best_params[0], best_params[1])

    return best_params
