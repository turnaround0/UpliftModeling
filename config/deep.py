from models import model_dta_deep

config = {
    'dataset': {
        'hillstrom': {
            'dta': {},
            'dta_deep': {'model': model_dta_deep, 'params': {'method': 'logistic'}},
        },
        # 'lalonde': {
        #     'dta': {},
        #     'dta_deep': {'model': model_dta_deep, 'params': {'method': 'linear'}},
        # },
        'criteo': {
            'dta': {},
            'dta_deep': {'model': model_dta_deep, 'params': {'method': 'logistic'}},
        },
    },
    'wrapper': False,
    'niv': True,
    'tune': False,
}
