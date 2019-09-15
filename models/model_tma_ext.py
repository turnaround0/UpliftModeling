import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

ext_params = {}


def set_params(max_round, u_list, final_uplift):
    global ext_params

    ext_params = {
        'max_round': max_round,
        'u_list': final_uplift + np.abs(final_uplift) * np.array(u_list),
    }
    print('Extraction params:', ext_params)


def get_predict_values(model_fit, data):
    if isinstance(model_fit[0], LinearRegression):
        pred_treat = model_fit[0].predict(data)
    else:
        pred_treat = model_fit[0].predict_proba(data)[:, 1]

    if isinstance(model_fit[1], LinearRegression):
        pred_control = model_fit[1].predict(data)
    else:
        pred_control = model_fit[1].predict_proba(data)[:, 1]

    return pred_treat, pred_control


def fit(x, y, t, method=LogisticRegression, **kwargs):
    fit_list = []
    rest = len(y)

    full_x = x.copy().reset_index(drop=True)
    full_y = y.copy().reset_index(drop=True)
    full_t = t.copy().reset_index(drop=True)
    x, y, t = full_x, full_y, full_t

    for idx in range(ext_params['max_round']):
        u_value = ext_params['u_list'][idx]
        if idx == ext_params['max_round'] - 1:
            treat_rows = (full_t == 1)
            control_rows = (full_t == 0)

            model_treat = method(**kwargs).fit(full_x[treat_rows], full_y[treat_rows])
            model_control = method(**kwargs).fit(full_x[control_rows], full_y[control_rows])

            fit_list.append((model_treat, model_control))

            ext_idx_list = full_y.index.tolist()
        else:
            treat_rows = (t == 1)
            control_rows = (t == 0)

            model_treat = method(**kwargs).fit(x[treat_rows], y[treat_rows])
            model_control = method(**kwargs).fit(x[control_rows], y[control_rows])

            pred_treat, pred_control = get_predict_values((model_treat, model_control), x)
            uplift = pred_treat - pred_control
            meet = pd.Series(uplift > u_value)
            ext_idx_list = meet[meet].index.tolist()

            x = x.drop(ext_idx_list).reset_index(drop=True)
            y = y.drop(ext_idx_list).reset_index(drop=True)
            t = t.drop(ext_idx_list).reset_index(drop=True)

            fit_list.append((model_treat, model_control))

        print('Round, rest, number of extraction:', idx, rest, len(ext_idx_list))
        rest -= len(ext_idx_list)

    return fit_list


def predict(obj, newdata, **kwargs):
    meet_list = []
    final_pred_treat, final_pred_control = None, None
    rest = len(newdata)
    for idx, model_fit in enumerate(obj):
        u_value = ext_params['u_list'][idx]

        pred_treat, pred_control = get_predict_values(model_fit, newdata)
        meet = pd.Series(pred_treat - pred_control > u_value)

        pred_treat = pd.Series(pred_treat)
        pred_control = pd.Series(pred_control)

        if idx == 0:
            final_pred_treat = pred_treat
            final_pred_control = pred_control
            final_pred_treat[~meet] = None
            final_pred_control[~meet] = None
        else:
            for prev_idx in range(idx):
                prev_meet = meet_list[prev_idx]
                meet[prev_meet] = False
            final_pred_treat[meet] = pred_treat[meet]
            final_pred_control[meet] = pred_control[meet]

        print('Round, rest, meet count:', idx, rest, meet.sum())
        meet_list.append(meet)
        rest -= meet.sum()

    return {
        'pr_y1_t1': final_pred_treat,
        'pr_y1_t0': final_pred_control,
    }
