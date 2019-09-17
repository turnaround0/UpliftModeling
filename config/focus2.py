import numpy as np

from models import model_dt_ed_focus2, model_dt_ed

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
            'dt_ed': {'model': model_dt_ed, 'params': params_tree_hillstrom},
            'dt_ed_focus2': {'model': model_dt_ed_focus2, 'params': params_tree_hillstrom, 'p_value': 0.38},
        },
        'criteo': {
            'dt_ed': {'model': model_dt_ed, 'params': params_tree_criteo},
            'dt_ed_focus2': {'model': model_dt_ed_focus2, 'params': params_tree_criteo, 'p_value': 0.39},
        },
    },
    'wrapper': False,
    'niv': True,
    'tune': False,
}
