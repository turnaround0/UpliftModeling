from models import model_dta_deep, model_deep

config = {
    'dataset': {
        'hillstrom': {
            'dta': {},
            'deep': {'model': model_dta_deep,
                     'params': {'method': 'logistic', 'lr': 1e-5, 'epochs': 50,
                                'batch_size': 256, 'decay': 1e-4}},
            'dta_deep': {'model': model_dta_deep,
                         'params': {'method': 'logistic', 'lr': 1e-5, 'epochs': 50,
                                    'batch_size': 256, 'decay': 1e-4}},
        },
        'lalonde': {
             'dta': {},
             'deep': {'model': model_deep,
                      'params': {'method': 'linear', 'lr': 1e-5, 'epochs': 300,
                                 'batch_size': 64, 'decay': 1e-4}},
             'dta_deep': {'model': model_dta_deep,
                          'params': {'method': 'linear', 'lr': 1e-5, 'epochs': 300,
                                     'batch_size': 64, 'decay': 1e-4}},
        },
        'criteo': {
            'dta': {},
            'deep': {'model': model_dta_deep,
                     'params': {'method': 'logistic', 'lr': 1e-5, 'epochs': 150,
                                'batch_size': 64, 'decay': 1e-4}},
            'dta_deep': {'model': model_dta_deep,
                         'params': {'method': 'logistic', 'lr': 1e-5, 'epochs': 150,
                                    'batch_size': 64, 'decay': 1e-4}},
        },
    },
    'wrapper': False,
    'niv': True,
    'tune': False,
}
