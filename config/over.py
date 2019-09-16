from sklearn.linear_model import LogisticRegression

from over import simple, smote, gan

params_logistic = {
    'method': LogisticRegression,
    'solver': 'newton-cg',
    'penalty': 'none',
}

config = {
    'dataset': {
        'hillstrom': {
            'tma': {'params': params_logistic},
            # 'tma_simple': {'over_sampling': simple.over_sampling, 'params': params_logistic},
            'tma_smote': {'over_sampling': smote.over_sampling, 'params': params_logistic},
            'tma_gan': {'over_sampling': gan.over_sampling, 'params': params_logistic},
            'tma_gan2': {'over_sampling': gan.over_sampling2, 'params': params_logistic},
        },
        'lalonde': {
            'tma': {},
            # 'tma_simple': {'over_sampling': simple.over_sampling},
            'tma_smote': {'over_sampling': smote.over_sampling},
        },
        'criteo': {
            'tma': {'params': params_logistic},
            # 'tma_simple': {'over_sampling': simple.over_sampling, 'params': params_logistic},
            'tma_smote': {'over_sampling': smote.over_sampling, 'params': params_logistic},
            'tma_gan': {'over_sampling': gan.over_sampling, 'params': params_logistic},
            'tma_gan2': {'over_sampling': gan.over_sampling2, 'params': params_logistic},
        },
    },
    'wrapper': False,
    'niv': True,
    'tune': False,
}
