import numpy as np
import pandas as pd
from models import model_dt

ext_params = {}


def set_params(max_round, u_list, final_uplift):
    global ext_params

    ext_params = {
        'max_round': max_round,
        'u_list': final_uplift + np.abs(final_uplift) * np.array(u_list),
    }
    print('Extraction params:', ext_params)


def fit(x, y, t, **kwargs):
    kwargs.update({'method': 'ed'})
    fit_list = []
    rest = len(y)

    full_x, full_y, full_t = x, y, t
    for idx in range(ext_params['max_round']):
        ext_idx_list = []
        u_value = ext_params['u_list'][idx]
        kwargs.update({'u_value': u_value, 'ext_idx_list': ext_idx_list})
        if idx == ext_params['max_round'] - 1:
            fit_list.append(model_dt.fit(full_x, full_y, full_t, **kwargs))
        else:
            fit_list.append(model_dt.fit(x, y, t, **kwargs))
            x = x.drop(ext_idx_list)
            y = y.drop(ext_idx_list)
            t = t.drop(ext_idx_list)
            rest -= len(ext_idx_list)

        print('Train) Round, u value, rest, number of extraction:', idx, u_value, rest, len(ext_idx_list))

    return fit_list


def predict(obj, newdata, **kwargs):
    kwargs.update({'method': 'ed'})

    meet_list = []
    final_pred = None
    rest = len(newdata)
    for idx, model_fit in enumerate(obj):
        u_value = ext_params['u_list'][idx]
        pred = model_dt.predict(model_fit, newdata, **kwargs)
        meet = pd.Series(pred['pr_y1_t1'] - pred['pr_y1_t0'] > u_value)

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
        print('Prediction) Round, u value, rest, meet count:', idx, u_value, rest, meet.sum())

    return final_pred
