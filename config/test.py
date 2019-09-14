from config import default
from models import model_tma, model_dta, model_lai, model_glai, model_rvtu, model_rf_ed

config = {
    'dataset': {
        'hillstrom': {
            'tma': {'model': model_tma, 'space': default.search_space_logistic},
        },
    },
    'wrapper': False,
    'niv': False,
    'tune': False,
    'over_sampling': False,
    'feature_combination': False,
}
