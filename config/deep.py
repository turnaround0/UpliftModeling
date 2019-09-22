from models import model_dta_deep

config = {
    'dataset': {
        #'hillstrom': {
        #    'dta': {},
        #    'dta_deep': {'model': model_dta_deep, 'params': {'method': 'logistic'}},
        #},
        'lalonde': {
             'dta': {},
             'dta_deep': {'model': model_dta_deep,
                          'params': {'method': 'linear', 'lr': 1e-4, 'epochs': 1000, 'batch_size': 64}},
        },
        'criteo': {
            'dta': {},
            'dta_deep': {'model': model_dta_deep,
                         'params': {'method': 'logistic', 'lr': 1e-5, 'epochs': 1000, 'batch_size': 64}},
        },
    },
    'wrapper': False,
    'niv': True,
    'tune': False,
}
