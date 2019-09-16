import numpy as np
import pandas as pd
from tensorflow import keras

from tree.tree import num_class


def build_generator(data_dim, latent_dim):
    model = keras.Sequential([
        keras.layers.Dense(128, input_dim=latent_dim),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(32),
        keras.layers.LeakyReLU(),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(16),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(data_dim, activation='sigmoid')
    ])
    # model.summary()

    noise = keras.layers.Input(shape=(latent_dim,))
    data = model(noise)

    return keras.models.Model(noise, data)


def build_discriminator(data_dim):
    model = keras.Sequential([
        keras.layers.Dense(128, input_dim=data_dim),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(32),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(16),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(1, activation='sigmoid')
    ])
    # model.summary()

    data = keras.layers.Input(shape=(data_dim,))
    validity = model(data)

    return keras.models.Model(data, validity)


def build_gan_network(learning_rate, beta_1, data_dim, latent_dim):
    generator_optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=beta_1)
    discriminator_optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=beta_1)

    # Build and compile the discriminator
    discriminator = build_discriminator(data_dim)
    discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])

    # Build the generator
    generator = build_generator(data_dim, latent_dim)

    # The generator takes noise as input and generates data
    z = keras.layers.Input(shape=(latent_dim,))
    data = generator(z)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The discriminator takes generated images as input and determines validity
    validity = discriminator(data)

    # The combined model  (stacked generator and discriminator)
    # Trains the generator to fool the discriminator
    combined = keras.models.Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=generator_optimizer)

    return generator, discriminator, combined


def train(df, epochs, batch_size, latent_dim, generator, discriminator, combined):
    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    last_idx = (df.shape[0] // batch_size) * batch_size
    for epoch in range(epochs):
        for idx in range(0, df.shape[0], batch_size):
            if idx == last_idx:
                continue

            ## Train Discriminator
            idx_list = df.index[idx: idx + batch_size]
            selected_df = df.loc[idx_list]

            # Generate a batch of new data
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            generated_data = generator.predict(noise)

            # Train the discriminator
            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(selected_df, valid)
            d_loss_fake = discriminator.train_on_batch(generated_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            ## Train Generator
            # Train the generator (to have the discriminator label samples as valid)
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            discriminator.trainable = False
            g_loss = combined.train_on_batch(noise, valid)

            # Plot the progress
            if (idx % (batch_size * 100)) == 0 or idx == last_idx - batch_size:
                print("%d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                      (epoch, idx, d_loss[0], 100 * d_loss[1], g_loss))


# Make fake data for balance among TR, TN, CR, CN
def over_sampling(X, T, Y):
    # Hyper-parameters
    learning_rate = 0.0001
    beta1 = 0.5
    batch_size = 64
    epochs = 2
    latent_dim = 128

    binary_list = []
    multi_list = []
    for col in X.columns:
        count = X[col].drop_duplicates().count()
        if count <= 2:
            binary_list.append(col)
        else:
            multi_list.append(col)

    # Normalization for non-binary columns
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

        backup_X = pd.concat([backup_X, generated_df])
        backup_T = pd.concat([backup_T, pd.Series([t] * num_add)])
        backup_Y = pd.concat([backup_Y, pd.Series([y] * num_add)])

    return X, T, Y


# Make fake data with x5
def over_sampling2(X, T, Y):
    # Hyper-parameters
    learning_rate = 0.00001
    beta1 = 0.5
    batch_size = 64
    epochs = 2
    latent_dim = 128
    num_fake_data = len(X) * 5

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
