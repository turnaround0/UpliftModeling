import numpy as np
import pandas as pd
from models import model_dt

ext_params = {}
predict_option = 2


def set_params(u_value, uplift):
    global ext_params

    ext_params = {
        'u_value': uplift + np.abs(uplift) * u_value,
    }
    print('Extraction params:', ext_params)


def fit(x, y, t, **kwargs):
    kwargs.update({'method': 'ed'})

    train_idx_list = []
    kwargs.update({'u_value': ext_params['u_value'], 'ext_idx_list': train_idx_list})
    all_fit = model_dt.fit(x, y, t, **kwargs)

    print('Number of check samples:', len(train_idx_list), '/', len(t))

    x = x.loc[train_idx_list]
    y = y.loc[train_idx_list]
    t = t.loc[train_idx_list]

    select_fit = model_dt.fit(x, y, t, **kwargs)

    return all_fit, select_fit


def predict(obj, newdata, **kwargs):
    kwargs.update({'method': 'ed'})

    u_value = ext_params['u_value']
    all_fit, select_fit = obj

    if predict_option == 1:
        all_pred = model_dt.predict(all_fit, newdata, **kwargs)
        select_pred = model_dt.predict(select_fit, newdata, **kwargs)

        meet = pd.Series(select_pred['pr_y1_t1'] - select_pred['pr_y1_t0'] > u_value)
        pred = all_pred
        pred[meet] = select_pred[meet]

        print('Number of meet samples:', meet.sum(), '/', len(meet))
        return pred
    elif predict_option == 2:
        return model_dt.predict(select_fit, newdata, **kwargs)
    else:
        print('Prediction option is wrong.')
        assert()
