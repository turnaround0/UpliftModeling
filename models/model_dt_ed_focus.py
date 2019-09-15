import numpy as np
import pandas as pd
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
    model_dt.fit(x, y, t, **kwargs)

    print('Number of check samples:', len(train_idx_list), '/', len(t))

    x = x.loc[train_idx_list]
    y = y.loc[train_idx_list]
    t = t.loc[train_idx_list]

    return model_dt.fit(x, y, t, **kwargs)


def predict(obj, newdata, **kwargs):
    kwargs.update({'method': 'ed'})
    return model_dt.predict(obj, newdata, **kwargs)
