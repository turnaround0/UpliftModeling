from config import default
from models import model_tma, model_dta, model_lai, model_glai, model_rvtu, \
    model_rf_ed, model_rf_kl, model_rf_chisq, model_rf_int

config = {
    'dataset': {
        'hillstrom': {
            'tma': {'model': model_tma, 'space': default.search_space_logistic,
                    'params': default.params_logistic},
            'dta': {'model': model_dta, 'space': default.search_space_logistic,
                    'params': default.params_logistic},
            'lai': {'model': model_lai, 'space': default.search_space_logistic,
                    'params': default.params_logistic},
            'glai': {'model': model_glai, 'space': default.search_space_logistic,
                     'params': default.params_logistic},
            'trans': {'model': model_rvtu, 'space': default.search_space_logistic,
                      'params': default.params_logistic},
            'urf_ed': {'model': model_rf_ed, 'space': default.search_space_tree_hillstrom,
                       'params': default.params_tree_hillstrom},
            'urf_kl': {'model': model_rf_kl, 'space': default.search_space_tree_hillstrom,
                       'params': default.params_tree_hillstrom},
            'urf_chisq': {'model': model_rf_chisq, 'space': default.search_space_tree_hillstrom,
                          'params': default.params_tree_hillstrom},
            'urf_int': {'model': model_rf_int, 'space': default.search_space_tree_hillstrom,
                        'params': default.params_tree_hillstrom},
        },
        'lalonde': {
            'tma': {'model': model_tma, 'space': default.search_space_linear,
                    'params': default.params_linear},
            'dta': {'model': model_tma, 'space': default.search_space_linear,
                    'params': default.params_linear},
            'urf_ed': {'model': model_rf_ed, 'space': default.search_space_tree_lalonde,
                       'params': default.params_tree_lalonde},
            'urf_kl': {'model': model_rf_kl, 'space': default.search_space_tree_lalonde,
                       'params': default.params_tree_lalonde},
            'urf_chisq': {'model': model_rf_chisq, 'space': default.search_space_tree_lalonde,
                          'params': default.params_tree_lalonde},
        },
        'criteo': {
            'tma': {'model': model_tma, 'space': default.search_space_logistic,
                    'params': default.params_logistic},
            'dta': {'model': model_dta, 'space': default.search_space_logistic,
                    'params': default.params_logistic},
            'lai': {'model': model_lai, 'space': default.search_space_logistic,
                    'params': default.params_logistic},
            'glai': {'model': model_glai, 'space': default.search_space_logistic,
                     'params': default.params_logistic},
            'trans': {'model': model_rvtu, 'space': default.search_space_logistic,
                      'params': default.params_logistic},
            'urf_ed': {'model': model_rf_ed, 'space': default.search_space_tree_criteo,
                       'params': default.params_tree_criteo},
            'urf_kl': {'model': model_rf_kl, 'space': default.search_space_tree_criteo,
                       'params': default.params_tree_criteo},
            'urf_chisq': {'model': model_rf_chisq, 'space': default.search_space_tree_criteo,
                          'params': default.params_tree_criteo},
            'urf_int': {'model': model_rf_int, 'space': default.search_space_tree_criteo,
                        'params': default.params_tree_criteo},
        },
    },
    'niv': True,
    'tune': True,
}
