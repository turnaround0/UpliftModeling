import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression


search_space_logistic = {
    'method': [LogisticRegression],
    'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
    'penalty': ['none', 'l2'],
    'tol': [1e-2, 1e-3, 1e-4],
    'C': [1e6, 1e3, 1, 1e-3, 1e-6],
}
search_space_linear = {
    'method': [LinearRegression],
}

search_space_tree_hillstrom = {
    'ntree': [10, ],
    'bagging_fraction': [0.6, ],
    'max_depth': [10, 5, ],
    'min_split': [2000, 1000, 500],
    # 'mtry': [3, ],  default = sqrt(#col)
    # 'min_bucket_t0': [100, ],  default = min_split / 4
    # 'min_bucket_t1': [100, ],  default = min_split / 4
}
s = 2000.
lalonde_bins = np.arange(-s, s * 10, s)
lalonde_bins[0] = -float('inf')
lalonde_bins[-1] = float('inf')
search_space_tree_lalonde = {
    'ntree': [10, ],
    'bagging_fraction': [0.6, ],
    'max_depth': [10, ],
    'min_split': [20, 10, 5, ],
    'bins': [lalonde_bins],
}
search_space_tree_criteo = {
    'ntree': [10, ],
    'bagging_fraction': [0.6, ],
    'max_depth': [10, ],
    'min_split': [50, 20, 5],
}

# Default parameters when not searching spaces
params_logistic = {
    'method': LogisticRegression,
    'solver': 'newton-cg',
    'penalty': 'l2',
    'tol': 1e-2,
    'C': 1,
}
params_linear = {
    'method': LinearRegression,
}
params_tree_hillstrom = {
    'ntree': 10,
    'bagging_fraction': 0.6,
    'max_depth': 10,
    'min_split': 1000,
}
params_tree_lalonde = {
    'ntree': 10,
    'bagging_fraction': 0.6,
    'max_depth': 10,
    'min_split': 10,
    'bins': lalonde_bins,
}
params_tree_criteo = {
    'ntree': 10,
    'bagging_fraction': 0.6,
    'max_depth': 10,
    'min_split': 20,
}
