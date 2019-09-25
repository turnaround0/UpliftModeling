import numpy as np
from tensorflow import keras


def build_generator(data_dim, latent_dim):
    model = keras.Sequential([
        keras.layers.Dense(64, input_dim=latent_dim, kernel_initializer='he_normal', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(64, kernel_initializer='he_normal', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(64, kernel_initializer='he_normal', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, kernel_initializer='he_normal', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, kernel_initializer='he_normal', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, kernel_initializer='he_normal', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(16, kernel_initializer='he_normal', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(16, kernel_initializer='he_normal', activation='relu'),
        keras.layers.Dense(data_dim, activation='sigmoid', kernel_initializer='he_normal'),
    ])
    # model.summary()

    noise = keras.layers.Input(shape=(latent_dim,))
    data = model(noise)

    return keras.models.Model(noise, data)


def build_discriminator(data_dim):
    model = keras.Sequential([
        keras.layers.Dense(64, input_dim=data_dim, kernel_initializer='he_normal', activation='relu'),
        keras.layers.Dense(64, kernel_initializer='he_normal', activation='relu'),
        keras.layers.Dense(64, kernel_initializer='he_normal', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, kernel_initializer='he_normal', activation='relu'),
        keras.layers.Dense(32, kernel_initializer='he_normal', activation='relu'),
        keras.layers.Dense(32, kernel_initializer='he_normal', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(16, kernel_initializer='he_normal', activation='relu'),
        keras.layers.Dense(16, kernel_initializer='he_normal', activation='relu'),
        keras.layers.Dense(1, activation='sigmoid', kernel_initializer='he_normal'),
    ])
    # model.summary()

    data = keras.layers.Input(shape=(data_dim,))
    validity = model(data)

    return keras.models.Model(data, validity)


def build_gan_network(gen_lr, dis_lr, data_dim, latent_dim):
    generator_optimizer = keras.optimizers.Adam(lr=gen_lr)
    discriminator_optimizer = keras.optimizers.Adam(lr=dis_lr)

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
