import tensorflow as tf
from tensorflow import keras


def build_model(input_len, output_len, activation):
    return keras.Sequential([
        keras.layers.Dense(64, input_dim=input_len, kernel_initializer='he_normal', activation='relu'),
        keras.layers.Dense(64, kernel_initializer='he_normal', activation='relu'),
        keras.layers.Dense(64, kernel_initializer='he_normal', activation='relu'),
        keras.layers.Dense(32, kernel_initializer='he_normal', activation='relu'),
        keras.layers.Dense(32, kernel_initializer='he_normal', activation='relu'),
        keras.layers.Dense(32, kernel_initializer='he_normal', activation='relu'),
        keras.layers.Dense(16, kernel_initializer='he_normal', activation='relu'),
        keras.layers.Dense(16, kernel_initializer='he_normal', activation='relu'),
        keras.layers.Dense(output_len, activation=activation, kernel_initializer='he_normal'),
    ])


def set_optimizer(model, lr, activation, decay):
    adam = keras.optimizers.Adam(lr=lr, decay=decay)
    if activation == 'linear':
        model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mse'])
    else:
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()

    return model


def init_seed(seed):
    tf.compat.v1.set_random_seed(seed)
