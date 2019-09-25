from models import model_dis
from over import gan

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
params_gan_criteo = {
    'gen_lr': 1e-4,
    'dis_lr': 3e-5,
    'batch_size': 64,
    'epochs': 100,
    'noise_size': 64,
    'major_multiple': 1.5,
    'minor_ratio': 0.5,
    'loss_type': 'vanilla',
}

config = {
    'dataset': {
        'hillstrom': {
            'dta': {},
            'dis': {'model': model_dis, 'over_sampling': gan.over_sampling,
                    'params_over': params_gan_hillstrom, 'params': {}},
        },
        'criteo': {
            # 'dta': {},
            'dis': {'model': model_dis, 'over_sampling': gan.over_sampling,
                    'params_over': params_gan_criteo, 'params': {}},
        },
    },
    'wrapper': False,
    'niv': True,
    'tune': False,
}
