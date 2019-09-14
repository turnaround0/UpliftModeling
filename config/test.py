from config import default
from models import model_tma, model_dta, model_lai, model_glai, model_rvtu, model_rf_ed, model_dt_ed
from over import simple, smote, gan

config = {
    'dataset': {
        'hillstrom': {
            'tma': {},
            # 'dt_ed': {'model': model_dt_ed},
            # 'urf_ed': {},
        },
    },
    'wrapper': False,
    'niv': False,
    'tune': False,
    'over_sampling': smote.over_sampling,
}
