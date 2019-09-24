import numpy as np
import pandas as pd
from models import model_dt

ext_params = {}


def set_params(max_round, p_value):
    global ext_params

    ext_params = {
        'max_round': max_round,
        'p_value': p_value,
        'u_list': [],
    }
    print('Extraction params:', ext_params)


def fit(x, y, t, **kwargs):
    kwargs.update({'method': 'ed'})
    fit_list = []
    rest = len(y)

    full_x, full_y, full_t = x, y, t
    for idx in range(ext_params['max_round']):
        ext_list = []
        p_value = ext_params['p_value']
        kwargs.update({'ext_list': ext_list})

        if idx == ext_params['max_round'] - 1:
            if idx == 0:
                fit_list.append(model_dt.fit(full_x, full_y, full_t, **kwargs))
            else:
                fit_list.append(fit_list[0])
            ext_idx_list = x.index.tolist()
            u_value = 0
        else:
            fit_list.append(model_dt.fit(x, y, t, **kwargs))
            df_ext = pd.DataFrame(ext_list).sort_values('abs_uplift', ascending=False)

            if len(df_ext) == 1:
                # If there is only one group after building tree, it should be halted.
                u_value = 0
                fit_list.pop()
                fit_list.append(fit_list[0])
                ext_idx_list = x.index.tolist()
                ext_params['u_list'].append(u_value)
                print('Before max round, tree has only one group.')
                print('Train) Round, u value, rest, number of extraction:', idx, u_value, rest, len(ext_idx_list))
                break

            df_ext['n_cumsum_samples'] = df_ext['n_samples'].cumsum()
            cut_len = rest * p_value
            cut_len_upper = df_ext[df_ext['n_cumsum_samples'] > cut_len]['n_cumsum_samples'].iloc[0]
            df_cut_ext = df_ext[df_ext['n_cumsum_samples'] <= cut_len_upper]
            if len(df_cut_ext) == len(df_ext):
                # Should not extract all data from training set
                df_cut_ext = df_ext.iloc[: -1]
            u_value = df_cut_ext.iloc[-1]['abs_uplift']

            ext_idx_list = df_cut_ext['idx_list'].sum()
            x = x.drop(ext_idx_list)
            y = y.drop(ext_idx_list)
            t = t.drop(ext_idx_list)
            rest -= len(ext_idx_list)

        ext_params['u_list'].append(u_value)
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
        meet = pd.Series(np.abs(pred['pr_y1_t1'] - pred['pr_y1_t0']) >= u_value)

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
