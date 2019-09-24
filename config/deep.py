from models import model_dta_deep, model_deep

config = {
    'dataset': {
        'hillstrom': {
            'dta': {},
            'deep': {'model': model_dta_deep,
                     'params': {'method': 'logistic', 'lr': 1e-3, 'epochs': 100,
                                'batch_size': 256, 'decay': 1e-2}},
            'dta_deep': {'model': model_dta_deep,
                         'params': {'method': 'logistic', 'lr': 1e-3, 'epochs': 100,
                                    'batch_size': 256, 'decay': 1e-2}},
        },
        'lalonde': {
             'dta': {},
             'deep': {'model': model_deep,
                      'params': {'method': 'linear', 'lr': 3e-4, 'epochs': 100,
                                 'batch_size': 64, 'decay': 3e-3}},
             'dta_deep': {'model': model_dta_deep,
                          'params': {'method': 'linear', 'lr': 3e-4, 'epochs': 100,
                                     'batch_size': 64, 'decay': 3e-3}},
        },
        'criteo': {
            'dta': {},
            'deep': {'model': model_dta_deep,
                     'params': {'method': 'logistic', 'lr': 7e-5, 'epochs': 150,
                                'batch_size': 64, 'decay': 7e-3}},
            'dta_deep': {'model': model_dta_deep,
                         'params': {'method': 'logistic', 'lr': 7e-5, 'epochs': 150,
                                    'batch_size': 64, 'decay': 7e-3}},
        },
    },
    'wrapper': False,
    'niv': True,
    'tune': False,
}
