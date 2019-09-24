import numpy as np
import pandas as pd

from deep.dnn import init_seed
from deep.gan import build_gan_network, train
from utils.utils import num_class, normalize, denormalize


def over_sampling(X, T, Y, params_over):
    gen_lr = params_over['gen_lr']
    dis_lr = params_over['dis_lr']
    beta1 = params_over['beta1']
    batch_size = params_over['batch_size']
    epochs = params_over['epochs']
    latent_dim = params_over['noise_size']
    seed = 1234
    max_loop = 2
    multiply_samples = 2

    init_seed(seed)

    X, T, Y = X.copy(), T.copy(), Y.copy()
    backup_X, backup_Y, backup_T = X.copy(), Y.copy(), T.copy()

    df = pd.concat([X, T, Y], axis=1)

    n_samples_list = pd.Series(num_class(df, 'Y', 'T'))
    target_samples = n_samples_list.max() * 2
    n_target_samples_list = target_samples - n_samples_list
    print('n_target_samples_list:', n_target_samples_list)

    df, normalize_vars = normalize(df)
    data_dim = df.shape[1]

    generator, discriminator, combined = \
        build_gan_network(gen_lr, dis_lr, beta1, data_dim, latent_dim)
    train(df, epochs, batch_size, latent_dim, generator, discriminator, combined)

    for _ in range(max_loop):
        num_fake_data = n_target_samples_list.sum() * multiply_samples
        print('num_fake_data:', num_fake_data)

        noise = np.random.normal(0, 1, (num_fake_data, latent_dim))
        generated_data = generator.predict(noise)
        generated_df = pd.DataFrame(generated_data, columns=df.columns)
        generated_df = denormalize(generated_df, normalize_vars)
        generated_df = generated_df.round()

        tr, tn, cr, cn = num_class(generated_df, 'Y', 'T')
        print('Generated data (tr, tn, cr, cn):', tr, tn, cr, cn)

        backup_X = pd.concat([backup_X, generated_df.drop(['T', 'Y'], axis=1)])
        backup_T = pd.concat([backup_T, generated_df['T']])
        backup_Y = pd.concat([backup_Y, generated_df['Y']])

    return backup_X, backup_T, backup_Y
