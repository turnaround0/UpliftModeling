from models import model_dta_deep, model_deep

config = {
    'dataset': {
        'hillstrom': {
            'dta': {},
            'deep': {'model': model_dta_deep,
                     'params': {'method': 'logistic', 'lr': 1e-5, 'epochs': 50,
                                'batch_size': 64, 'decay': 0.01}},
            'dta_deep': {'model': model_dta_deep,
                         'params': {'method': 'logistic', 'lr': 1e-5, 'epochs': 50,
                                    'batch_size': 64, 'decay': 0.01}},
        },
        'lalonde': {
             'dta': {},
             'deep': {'model': model_deep,
                      'params': {'method': 'linear', 'lr': 1e-5, 'epochs': 300,
                                 'batch_size': 64, 'decay': 0.01}},
             'dta_deep': {'model': model_dta_deep,
                          'params': {'method': 'linear', 'lr': 1e-5, 'epochs': 300,
                                     'batch_size': 64, 'decay': 0.01}},
        },
        'criteo': {
            'dta': {},
            'deep': {'model': model_dta_deep,
                     'params': {'method': 'logistic', 'lr': 1e-5, 'epochs': 100,
                                'batch_size': 64, 'decay': 0.01}},
            'dta_deep': {'model': model_dta_deep,
                         'params': {'method': 'logistic', 'lr': 1e-5, 'epochs': 100,
                                    'batch_size': 64, 'decay': 0.01}},
        },
    },
    'wrapper': False,
    'niv': True,
    'tune': False,
}
