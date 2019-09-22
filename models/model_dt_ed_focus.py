import pandas as pd
from models import model_dt

ext_params = {}


def set_params(p_value):
    global ext_params

    ext_params = {
        'p_value': p_value,
    }
    print('Extraction params:', ext_params)


def fit(x, y, t, **kwargs):
    kwargs.update({'method': 'ed'})

    train_list = []
    kwargs.update({'ext_list': train_list})
    all_fit = model_dt.fit(x, y, t, **kwargs)

    cut_len = len(y) * (1 - ext_params['p_value'])
    df_ext = pd.DataFrame(train_list).sort_values('abs_uplift', ascending=False)
    df_ext['n_cumsum_samples'] = df_ext['n_samples'].cumsum()
    df_cut_ext = df_ext[df_ext['n_cumsum_samples'] > cut_len]
    train_idx_list = df_cut_ext['idx_list'].sum()

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
