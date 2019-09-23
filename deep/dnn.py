from tensorflow import keras


def build_model(input_len, output_len, lr, activation):
    model = keras.Sequential([
        keras.layers.Dense(128, input_shape=(input_len,), kernel_initializer='he_normal'),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(128, kernel_initializer='he_normal'),
        keras.layers.LeakyReLU(),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(64, kernel_initializer='he_normal'),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(64, kernel_initializer='he_normal'),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(64, kernel_initializer='he_normal'),
        keras.layers.LeakyReLU(),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(32, kernel_initializer='he_normal'),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(32, kernel_initializer='he_normal'),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(16, kernel_initializer='he_normal'),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(8, kernel_initializer='he_normal'),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(output_len, activation=activation, kernel_initializer='he_normal'),
    ])

    adam = keras.optimizers.Adam(lr=lr)
    if activation == 'linear':
        model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mse'])
    else:
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()

    return model
