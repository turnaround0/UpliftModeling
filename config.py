from sklearn.linear_model import LogisticRegression, LinearRegression

from models import model_tma, model_dta, model_lai, model_glai, model_rvtu, model_rf

models = {
    #'tma': model_tma,
    #'dta': model_dta,
    #'lai': model_lai,
    'glai': model_glai,
    #'trans': model_rvtu,
    #'urf_ed': model_rf,
    #'urf_kl': model_rf,
    #'urf_chisq': model_rf,
    #'urf_int': model_rf,
}
lalonde_models = {
    'tma': model_tma,
    'dta': model_dta,
    'urf_ed': model_rf,
    'urf_kl': model_rf,
    'urf_chisq': model_rf,
    'urf_int': model_rf,
}
search_space = {
    'method': [LogisticRegression],
    'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
    'penalty': ['none', 'l2'],
    'tol': [1e-2, 1e-3, 1e-4],
    'C': [1e6, 1e3, 1, 1e-3, 1e-6],
    'max_iter': [2],
}
search_space_for_linear = {
    'method': [LinearRegression],
}
search_space_for_tree = {
    'ntree': [10, 20, ],
    'mtry': [3, ],
    'bagging_fraction': [0.6, ],
    # 'method': ['ED', ],
    'max_depth': [10, 5, ],
    'min_split': [2000, 1000, ],
    # 'min_bucket_t0': [100, ],
    # 'min_bucket_t1': [100, ],
}
search_space_for_dta = {
    'method': [LogisticRegression],
    'solver': ['liblinear', ],
}
urf_methods = {
    'urf_ed': 'ed',
    'urf_kl': 'kl',
    'urf_chisq': 'chisq',
    'urf_int': 'int'
}
wrapper_models = ['tma', 'dta', 'trans']


def get_models(dataset_name):
    return {
        'hillstrom': models,
        'lalonde': lalonde_models,
        'criteo': models,
    }[dataset_name]


def get_search_space(dataset_name, model_name):
    if model_name in ['tma', 'trans', 'dta', 'lai', 'glai']:
        if dataset_name == 'lalonde':
            return search_space_for_linear
        else:
            return search_space
    elif model_name == 'dta':
        return search_space_for_dta
    elif model_name.startswith('urf'):
        return search_space_for_tree
    else:
        return {}
