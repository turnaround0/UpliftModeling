from models import model_dis
from over import gan

params_tfgan_hillstrom = {
    'gen_lr': 7e-4,
    'dis_lr': 2e-5,
    'batch_size': 256,
    'epochs': 100,
    'noise_size': 64,
    'major_multiple': 0,
    'minor_ratio': 0,
    'loss_type': 'vanilla',
}
params_tfgan_criteo = {
    'gen_lr': 1e-4,
    'dis_lr': 3e-5,
    'batch_size': 64,
    'epochs': 200,
    'noise_size': 64,
    'major_multiple': 0,
    'minor_ratio': 0,
    'loss_type': 'vanilla',
}

config = {
    'dataset': {
        'hillstrom': {
            'dta': {},
            'dis': {'model': model_dis, 'over_sampling': gan.over_sampling,
                    'params_over': params_tfgan_hillstrom, 'params': {}},
        },
        'criteo': {
            'dta': {},
            'dis': {'model': model_dis, 'over_sampling': gan.over_sampling,
                    'params_over': params_tfgan_criteo, 'params': {}},
        },
    },
    'wrapper': False,
    'niv': True,
    'tune': False,
}
