import pandas as pd
from models import model_dt

ext_params = {}


def set_params(max_round, p_list):
    global ext_params

    ext_params = {
        'max_round': max_round,
        'p_list': p_list,
    }
    print('Extraction params:', ext_params)


def fit(x, y, t, **kwargs):
    kwargs.update({'method': 'ed'})
    fit_list = []
    rest = len(y)

    full_x, full_y, full_t = x, y, t
    for idx in range(ext_params['max_round']):
        ext_idx_list = []
        p_value = ext_params['p_list'][idx]
        kwargs.update({'p_value': p_value, 'ext_idx_list': ext_idx_list})
        if idx == ext_params['max_round'] - 1:
            fit_list.append(model_dt.fit(full_x, full_y, full_t, **kwargs))
        else:
            if rest == 0:
                fit_list.append(None)
                continue

            fit_list.append(model_dt.fit(x, y, t, **kwargs))
            x = x.drop(ext_idx_list)
            y = y.drop(ext_idx_list)
            t = t.drop(ext_idx_list)
            rest -= len(ext_idx_list)

        print('Train) Round, p value, rest, number of extraction:', idx, p_value, rest, len(ext_idx_list))

    return fit_list


def predict(obj, newdata, **kwargs):
    kwargs.update({'method': 'ed'})

    meet_list = []
    final_pred = None
    rest = len(newdata)
    for idx, model_fit in enumerate(obj):
        if model_fit is None:
            meet = meet_list[idx - 1].copy()
            meet[:] = False
            meet_list.append(meet)
            continue

        p_value = ext_params['p_list'][idx]
        pred = model_dt.predict(model_fit, newdata, **kwargs)
        meet = pd.Series(((pred['pr_y1_t1'] - 0.5).abs() >= p_value) |
                         ((pred['pr_y1_t0'] - 0.5).abs() >= p_value))

        if idx == 0:
            final_pred = pred
            final_pred[~meet] = None
        else:
            for prev_idx in range(idx):
                prev_meet = meet_list[prev_idx]
                meet[prev_meet] = False
            final_pred[meet] = pred[meet]

        meet_list.append(meet)
        if idx == ext_params['max_round'] - 1:
            rest = 0
        else:
            rest -= meet.sum()
        print('Prediction) Round, p value, rest, meet count:', idx, p_value, rest, meet.sum())

    return final_pred
