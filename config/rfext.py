import numpy as np

from models import model_dt_ed_ext, model_dt_ed, model_rf_ed_ext, model_rf_ed

params_tree_hillstrom = {
    'max_depth': 10,
    'min_split': 500,
}
s = 2000.
lalonde_bins = np.arange(-s, s * 10, s)
lalonde_bins[0] = -float('inf')
lalonde_bins[-1] = float('inf')
params_tree_lalonde = {
    'max_depth': 10,
    'min_split': 10,
    'bins': lalonde_bins,
}
params_tree_criteo = {
    'max_depth': 10,
    'min_split': 20,
}

config = {
    'dataset': {
        'hillstrom': {
            # 'dt_ed': {'model': model_dt_ed, 'params': params_tree_hillstrom},
            # 'dt_ed_ext': {'model': model_dt_ed_ext, 'params': params_tree_hillstrom,
            #               'max_round': 4, 'p_value': 0.2},
            'urf_ed': {'model': model_rf_ed, 'params': params_tree_hillstrom},
            'urf_ed_ext': {'model': model_rf_ed_ext, 'params': params_tree_hillstrom,
                           'max_round': 3, 'p_value': 0.05},
        },
        'lalonde': {
            # 'dt_ed': {'model': model_dt_ed, 'params': params_tree_lalonde},
            # 'dt_ed_ext': {'model': model_dt_ed_ext, 'params': params_tree_lalonde,
            #               'max_round': 7, 'p_value': 0.1},
            'urf_ed': {'model': model_rf_ed, 'params': params_tree_lalonde},
            'urf_ed_ext': {'model': model_rf_ed_ext, 'params': params_tree_lalonde,
                           'max_round': 4, 'p_value': 0.03},
        },
        'criteo': {
            # 'dt_ed': {'model': model_dt_ed, 'params': params_tree_criteo},
            # 'dt_ed_ext': {'model': model_dt_ed_ext, 'params': params_tree_criteo,
            #               'max_round': 7, 'p_value': 0.1},
            'urf_ed': {'model': model_rf_ed, 'params': params_tree_criteo},
            'urf_ed_ext': {'model': model_rf_ed_ext, 'params': params_tree_criteo,
                           'max_round': 4, 'p_value': 0.03},
        },
    },
    'wrapper': False,
    'niv': True,
    'tune': False,
}
