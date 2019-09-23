import numpy as np
import pandas as pd
import tensorflow as tf

from deep.gan import build_gan_network, train
from utils.utils import num_class


# Make fake data for balance among TR, TN, CR, CN
def over_sampling(X, T, Y):
    # Hyper-parameters
    learning_rate = 0.0001
    beta1 = 0.5
    batch_size = 64
    epochs = 30
    latent_dim = 128
    tf.compat.v1.set_random_seed(1234)

    binary_list = []
    multi_list = []
    for col in X.columns:
        count = X[col].drop_duplicates().count()
        if count <= 2:
            binary_list.append(col)
        else:
            multi_list.append(col)

    # Normalization for non-binary columns
    X = X.copy()
    T = T.copy()
    Y = Y.copy()
    backup_X = X.copy()
    backup_Y = Y.copy()
    backup_T = T.copy()
    for col in multi_list:
        max_val = X[col].max()
        min_val = X[col].min()
        X[col] = (X[col] - min_val) / (max_val - min_val)

    df = pd.concat([X, T, Y], axis=1)

    tr, tn, cr, cn = num_class(df, 'Y', 'T')
    list_num_class = [tr, tn, cr, cn]
    pair_class_list = [(1, 1), (1, 0), (0, 1), (0, 0)]
    num_max_class = max(list_num_class)

    generated_X_list = []
    generated_T_list = []
    generated_Y_list = []

    for idx in range(len(list_num_class)):
        num_add = num_max_class - list_num_class[idx]
        if num_add == 0:
            continue

        print('Training (T, Y):', pair_class_list[idx])

        data_dim = X.shape[1]
        t = pair_class_list[idx][0]
        y = pair_class_list[idx][1]

        sel_X = X[(df['T'] == t) & (df['Y'] == y)]

        generator, discriminator, combined = build_gan_network(learning_rate, beta1, data_dim, latent_dim)
        train(sel_X, epochs, batch_size, latent_dim, generator, discriminator, combined)

        noise = np.random.normal(0, 1, (num_add, latent_dim))
        generated_data = generator.predict(noise)
        generated_df = pd.DataFrame(generated_data, columns=X.columns)

        # Fix non-binary columns
        for col in multi_list:
            max_val = backup_X[col].max()
            min_val = backup_X[col].min()
            generated_df[col] = round(generated_df[col] * (max_val - min_val) + min_val)

        # Fix binary columns
        for col in binary_list:
            generated_df.loc[generated_df[col] >= 0.5, col] = 1
            generated_df.loc[generated_df[col] < 0.5, col] = 0

        # Append generated dataframe
        generated_X_list.append(generated_df)
        generated_T_list.append(pd.Series([t] * num_add, name='T'))
        generated_Y_list.append(pd.Series([y] * num_add, name='Y'))

    # Combine original and generated data
    X = pd.concat([backup_X] + generated_X_list)
    T = pd.concat([backup_T] + generated_T_list)
    Y = pd.concat([backup_Y] + generated_Y_list)

    return X, T, Y


# Make fake data with x5
def over_sampling2(X, T, Y):
    # Hyper-parameters
    learning_rate = 0.00001
    beta1 = 0.5
    batch_size = 64
    epochs = 30
    latent_dim = 128
    num_fake_data = len(X) * 5
    tf.compat.v1.set_random_seed(1234)

    binary_list = []
    multi_list = []
    for col in X.columns:
        count = X[col].drop_duplicates().count()
        if count <= 2:
            binary_list.append(col)
        else:
            multi_list.append(col)

    binary_list.append('T')
    binary_list.append('Y')

    # Normalization for non-binary columns
    X = X.copy()
    T = T.copy()
    Y = Y.copy()
    backup_X = X.copy()
    backup_Y = Y.copy()
    backup_T = T.copy()
    for col in multi_list:
        max_val = X[col].max()
        min_val = X[col].min()
        X[col] = (X[col] - min_val) / (max_val - min_val)

    df = pd.concat([X, T, Y], axis=1)

    data_dim = df.shape[1]

    generator, discriminator, combined = build_gan_network(learning_rate, beta1, data_dim, latent_dim)
    train(df, epochs, batch_size, latent_dim, generator, discriminator, combined)

    noise = np.random.normal(0, 1, (num_fake_data, latent_dim))
    generated_data = generator.predict(noise)
    generated_df = pd.DataFrame(generated_data, columns=df.columns)

    # Fix non-binary columns
    for col in multi_list:
        max_val = backup_X[col].max()
        min_val = backup_X[col].min()
        generated_df[col] = round(generated_df[col] * (max_val - min_val) + min_val)

    # Fix binary columns
    for col in binary_list:
        generated_df.loc[generated_df[col] >= 0.5, col] = 1
        generated_df.loc[generated_df[col] < 0.5, col] = 0

    tr, tn, cr, cn = num_class(generated_df, 'Y', 'T')
    print('Generated data (tr, tn, cr, cn):', tr, tn, cr, cn)

    X = pd.concat([backup_X, generated_df.drop(['T', 'Y'], axis=1)])
    T = pd.concat([backup_T, generated_df['T']])
    Y = pd.concat([backup_Y, generated_df['Y']])

    return X, T, Y
