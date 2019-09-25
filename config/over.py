from sklearn.linear_model import LogisticRegression

from over import smote, gan, tfgan

params_logistic = {
    'method': LogisticRegression,
    'solver': 'newton-cg',
    'penalty': 'none',
}
params_gan_hillstrom = {
    'gen_lr': 1e-4,
    'dis_lr': 7e-6,
    'batch_size': 256,
    'epochs': 70,
    'noise_size': 64,
    'major_multiple': 1.5,
    'minor_ratio': 0.5,
    'loss_type': 'vanilla',
}
params_tfgan_hillstrom = {
    'gen_lr': 1e-4,
    'dis_lr': 7e-6,
    'batch_size': 256,
    'epochs': 500,
    'noise_size': 64,
    'major_multiple': 1.5,
    'minor_ratio': 0.5,
}
params_gan_criteo = {
    'gen_lr': 1e-4,
    'dis_lr': 3e-5,
    'batch_size': 64,
    'epochs': 10,
    'noise_size': 64,
    'major_multiple': 1.5,
    'minor_ratio': 0.5,
    'loss_type': 'vanilla',
}
params_tfgan_criteo = {
    'gen_lr': 1e-4,
    'dis_lr': 3e-5,
    'batch_size': 64,
    'epochs': 3000,
    'noise_size': 64,
    'major_multiple': 1.5,
    'minor_ratio': 0.5,
}

config = {
    'dataset': {
        'hillstrom': {
            'dta': {'params': params_logistic},
            'dta_smote': {'over_sampling': smote.over_sampling, 'params': params_logistic},
            'dta_gan': {'over_sampling': gan.over_sampling, 'params': params_logistic,
                        'params_over': params_gan_hillstrom},
            'dta_tfgan': {'over_sampling': tfgan.over_sampling, 'params': params_logistic,
                          'params_over': params_tfgan_hillstrom},
        },
        'criteo': {
            'dta': {'params': params_logistic},
            'dta_smote': {'over_sampling': smote.over_sampling, 'params': params_logistic},
            'dta_gan': {'over_sampling': gan.over_sampling, 'params': params_logistic,
                        'params_over': params_gan_criteo},
            'dta_tfgan': {'over_sampling': tfgan.over_sampling, 'params': params_logistic,
                          'params_over': params_tfgan_criteo},
        },
    },
    'wrapper': False,
    'niv': True,
    'tune': False,
}
