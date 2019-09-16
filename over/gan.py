import numpy as np
import pandas as pd
from tensorflow import keras


def get_optimizer(learning_rate, beta_1):
    return keras.optimizers.Adam(lr=learning_rate, beta_1=beta_1)


def build_generator(input_dim, output_dim):
    model = keras.Sequential([
        keras.layers.Dense(64, input_dim=input_dim),
        keras.layers.LeakyReLU(0.2),

        keras.layers.Dense(32),
        keras.layers.LeakyReLU(0.2),

        keras.layers.Dense(output_dim, activation='sigmoid')
    ])
    model.summary()

    noise = keras.layers.Input(shape=(input_dim,))
    data = model(noise)

    return keras.models.Model(noise, data)


def build_discriminator(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(64, input_dim=input_dim),
        keras.layers.LeakyReLU(0.2),

        keras.layers.Dense(32),
        keras.layers.LeakyReLU(0.2),

        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()

    data = keras.layers.Input(shape=input_dim)
    validity = model(data)

    return keras.models.Model(data, validity)


def build_gan_network(learning_rate, beta_1, input_dim, output_dim):
    optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=beta_1)

    # Build and compile the discriminator
    discriminator = build_discriminator(input_dim)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Build the generator
    generator = build_generator(input_dim, output_dim)

    # The generator takes noise as input and generates data
    z = keras.layers.Input(shape=(input_dim,))
    data = generator(z)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The discriminator takes generated images as input and determines validity
    validity = discriminator(data)

    # The combined model  (stacked generator and discriminator)
    # Trains the generator to fool the discriminator
    combined = keras.models.Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    return generator, discriminator, combined


def train(df, epochs, batch_size, input_dim, generator, discriminator, combined):
    # TODO: Input normalization

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        ## Train Discriminator
        # Select a random batch of images
        idx = np.random.randint(0, df.shape[0], batch_size)
        selected_df = df.loc[idx]

        # Generate a batch of new data
        noise = np.random.normal(0, 1, (batch_size, input_dim))
        generated_data = generator.predict(noise)
        print('START gen')
        print(generated_data)
        print('END gen')

        # Train the discriminator
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(selected_df, valid)
        d_loss_fake = discriminator.train_on_batch(generated_data, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        ## Train Generator
        # Train the generator (to have the discriminator label samples as valid)
        noise = np.random.normal(0, 1, (batch_size, input_dim))
        discriminator.trainable = False
        g_loss = combined.train_on_batch(noise, valid)

        # Plot the progress
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))


def over_sampling(X, T, Y):
    # Hyper-parameters
    learning_rate = 0.0002
    beta1 = 0.5
    batch_size = 64
    epochs = 100

    df = pd.concat([X, T, Y], axis=1)

    input_dim = df.shape[1]
    output_dim = df.shape[1]

    keras.backend.clear_session()

    generator, discriminator, combined = build_gan_network(learning_rate, beta1, input_dim, output_dim)
    train(df, epochs, batch_size, input_dim, generator, discriminator, combined)

    return X, T, Y
