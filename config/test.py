from config import default
from models import model_tma, model_dta, model_lai, model_glai, model_rvtu, model_rf_ed, model_dt_ed

config = {
    'dataset': {
        'hillstrom': {
            'tma': {},
            'dt_ed': {'model': model_dt_ed, 'space': default.search_space_tree_hillstrom,
                      'params': default.params_tree_hillstrom},
            'urf_ed': {},
        },
    },
    'wrapper': False,
    'niv': False,
    'tune': False,
    'over_sampling': False,
    'feature_combination': False,
}
