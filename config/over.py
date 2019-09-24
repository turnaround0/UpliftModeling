from sklearn.linear_model import LogisticRegression

from over import smote, gan

params_logistic = {
    'method': LogisticRegression,
    'solver': 'newton-cg',
    'penalty': 'none',
}
params_gan_hillstrom = {
    'gen_lr': 1e-4,
    'dis_lr': 7e-6,
    'batch_size': 256,
    'epochs': 50,
    'noise_size': 64,
    'beta1': 0.5,
}
params_gan_criteo = {
    'gen_lr': 1e-4,
    'dis_lr': 3e-5,
    'batch_size': 64,
    'epochs': 100,
    'noise_size': 64,
    'beta1': 0.5,
}

config = {
    'dataset': {
        'hillstrom': {
            # 'dta': {'params': params_logistic},
            # 'dta_smote': {'over_sampling': smote.over_sampling, 'params': params_logistic},
            'dta_gan': {'over_sampling': gan.over_sampling, 'params': params_logistic,
                        'params_over': params_gan_hillstrom},
        },
        # 'criteo': {
        #     'dta': {'params': params_logistic},
        #     'dta_smote': {'over_sampling': smote.over_sampling, 'params': params_logistic},
        #     'dta_gan': {'over_sampling': gan.over_sampling, 'params': params_logistic,
        #                 'params_over': params_gan_criteo},
        # },
    },
    'wrapper': False,
    'niv': True,
    'tune': False,
}
