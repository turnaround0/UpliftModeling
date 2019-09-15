import numpy as np

from models import model_tma_ext, model_dt_ed_ext, model_dt_ed

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
        #'hillstrom': {
        #    'dt_ed': {'model': model_dt_ed, 'params': params_tree_hillstrom},
        #    'dt_ed_ext': {'model': model_dt_ed_ext, 'params': params_tree_hillstrom,
        #                  'max_round': 6, 'u_list': [1.5, 1.0, 0.7, 0.5, 0.3, -float('INF')]},
        #    # 'tma_ext': {'model': model_tma_ext},
        #    # 'tma': {},
        #},
        'lalonde': {
            'dt_ed': {'model': model_dt_ed, 'params': params_tree_lalonde},
            'dt_ed_ext': {'model': model_dt_ed_ext, 'params': params_tree_lalonde,
                          'max_round': 6, 'u_list': [1.5, 1.0, 0.7, 0.5, 0.3, -float('INF')]},
            # 'tma_ext': {'model': model_tma_ext},
            # 'tma': {},
        },
        #'criteo': {
        #    'dt_ed': {'model': model_dt_ed, 'params': params_tree_criteo},
        #    'dt_ed_ext': {'model': model_dt_ed_ext, 'params': params_tree_criteo,
        #                  'max_round': 4, 'u_list': [2.0, 1.0, 0.5, -float('INF')]},
        #    # 'tma_ext': {'model': model_tma_ext},
        #    # 'tma': {},
        #},
    },
    'wrapper': False,
    'niv': True,
    'tune': False,
}
