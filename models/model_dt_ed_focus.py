import numpy as np
from models import model_dt

ext_params = {}


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
    _, select_fit = obj

    return model_dt.predict(select_fit, newdata, **kwargs)
