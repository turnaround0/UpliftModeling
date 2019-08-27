import numpy as np
import pandas as pd
from experiment import performance, qini


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
        # print('    {}'.format(params))
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


def do_niv_variable_selection(x, y, t):
    """
    NIV variable selection procedure

    Args:
        x: predictor variables of training dataset,
        y: target variables of training dataset,
        t: treatment variables of training dataset,
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
    idx = 0
    for col in x.columns:
        df = pd.concat([x[col].rename(col), y1_t.rename('y1_t'), y0_t.rename('y0_t'),
                        y1_c.rename('y1_c'), y0_c.rename('y0_c')], axis=1)
        x_group = df.groupby(x[col])
        x_sum = x_group.sum()

        woe_t = np.log((x_sum['y1_t'] * sum_y0_t) / (x_sum['y0_t'] * sum_y1_t))
        woe_c = np.log((x_sum['y1_c'] * sum_y0_c) / (x_sum['y0_c'] * sum_y1_c))
        woe_t[x_sum['y1_t'] == 0] = 0
        woe_t[x_sum['y0_t'] == 0] = 0
        woe_c[x_sum['y1_c'] == 0] = 0
        woe_c[x_sum['y0_c'] == 0] = 0
        nwoe = woe_t - woe_c

        p_x_y1_t = x_sum['y1_t'] / sum_y1_t if sum_y1_t > 0 else 0
        p_x_y0_t = x_sum['y0_t'] / sum_y0_t if sum_y0_t > 0 else 0
        p_x_y1_c = x_sum['y1_c'] / sum_y1_c if sum_y1_c > 0 else 0
        p_x_y0_c = x_sum['y0_c'] / sum_y0_c if sum_y0_c > 0 else 0
        niv_weight = (p_x_y1_t * p_x_y0_c - p_x_y0_t * p_x_y1_c)

        niv_row = 100 * nwoe * niv_weight
        niv = niv_row.sum()
        niv_dict[col] = niv
        idx += 1
        if idx > 200:
            break

    s_niv = pd.Series(niv_dict)
    s_selected_niv = s_niv.sort_values(ascending=False)[: 12]

    return s_selected_niv.index
