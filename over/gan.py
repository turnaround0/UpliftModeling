import numpy as np
import pandas as pd

from deep.dnn import init_tf
from deep.gan import build_gan_network, train
from utils.utils import num_class, split_class, normalize, denormalize

stored_discriminator = None


def over_sampling(X, T, Y, params_over):
    gen_lr = params_over['gen_lr']
    dis_lr = params_over['dis_lr']
    batch_size = params_over['batch_size']
    epochs = params_over['epochs']
    latent_dim = params_over['noise_size']
    major_multiple = params_over['major_multiple']
    minor_ratio = params_over['minor_ratio']
    loss_type = params_over['loss_type']
    seed = 1234
    max_loop = 10
    fake_multiple = 5

    init_tf(seed)

    train_df = pd.concat([X, T, Y], axis=1).copy()
    out_df = train_df.copy()

    n_samples = pd.Series(num_class(train_df, 'Y', 'T'))
    print('Initial samples:', n_samples.tolist())

    num_major = n_samples.max() * major_multiple
    idx_major = n_samples.argmax()
    num_minor = num_major * minor_ratio

    n_rest_samples = [num_minor] * len(n_samples)
    n_rest_samples[idx_major] = num_major
    n_rest_samples = pd.Series(n_rest_samples).round().astype('int32')
    n_rest_samples -= n_samples
    n_rest_samples[n_rest_samples < 0] = 0
    num_fake_data = n_rest_samples.sum() * fake_multiple
    print('Initial rest samples:', n_rest_samples.tolist())

    train_df, normalize_vars = normalize(train_df)
    data_dim = train_df.shape[1]

    generator, discriminator, combined = \
        build_gan_network(gen_lr, dis_lr, data_dim, latent_dim, loss_type)
    train(train_df, epochs, batch_size, latent_dim, generator, discriminator, combined)

    global stored_discriminator
    stored_discriminator = discriminator

    for _ in range(max_loop):
        if n_rest_samples.sum() == 0:
            break
        noise = np.random.normal(0, 1, (num_fake_data, latent_dim))
        gen_data = generator.predict(noise)
        gen_df = pd.DataFrame(gen_data, columns=train_df.columns)
        gen_df = denormalize(gen_df, normalize_vars)
        gen_df = gen_df.round()

        tr, tn, cr, cn = num_class(gen_df, 'Y', 'T')
        print('Generated data (tr, tn, cr, cn):', tr, tn, cr, cn)

        gen_df_list = split_class(gen_df, 'Y', 'T')
        for idx, df in enumerate(gen_df_list):
            n_sel_samples = df.shape[0] if df.shape[0] < n_rest_samples[idx] else n_rest_samples[idx]
            n_rest_samples[idx] -= n_sel_samples
            sel_df = gen_df.iloc[: n_sel_samples]
            out_df = pd.concat([out_df, sel_df])

        print('Rest samples:', n_rest_samples.tolist())

    out_df = out_df.reset_index(drop=True).sample(frac=1)
    X = out_df.drop(['T', 'Y'], axis=1)
    T = out_df['T']
    Y = out_df['Y']

    return X, T, Y


def get_stored_discriminator():
    global stored_discriminator
    return stored_discriminator
