import pandas as pd
import tensorflow as tf
from tensorflow import keras
from utils.utils import normalize, denormalize

normalize_vars = None


def build_model(x_len, lr, activation):
    model = keras.Sequential([
        keras.layers.Dense(128, input_shape=(x_len,), kernel_initializer='he_normal'),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(128, kernel_initializer='normal'),
        keras.layers.LeakyReLU(),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(64, kernel_initializer='normal'),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(64, kernel_initializer='normal'),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(64, kernel_initializer='normal'),
        keras.layers.LeakyReLU(),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(32, kernel_initializer='normal'),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(32, kernel_initializer='normal'),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(16, kernel_initializer='normal'),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(8, kernel_initializer='normal'),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(1, activation=activation, kernel_initializer='normal'),
    ])

    adam = keras.optimizers.Adam(lr=lr)
    if activation == 'linear':
        model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mse'])
    else:
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()

    return model


def fit(x, y, t, **kwargs):
    lr = kwargs.get('lr')
    epochs = kwargs.get('epochs')
    batch_size = kwargs.get('batch_size')
    method = kwargs.get('method')
    activation = 'sigmoid' if method == 'logistic' else 'linear'

    tf.compat.v1.set_random_seed(1234)

    df = x.copy()
    df['treated'] = t

    df, _ = normalize(df)
    if method == 'linear':
        global normalize_vars
        y, normalize_vars = normalize(pd.DataFrame(y))
    else:
        normalize_vars = None

    model = build_model(df.shape[1], lr, activation)
    model.fit(df, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)

    return model


def predict(obj, newdata, **kwargs):
    df_treat = newdata.copy()
    df_control = newdata.copy()

    df_treat['treated'] = 1
    df_control['treated'] = 0

    pred_treat = [val[0] for val in obj.predict(df_treat)]
    pred_control = [val[0] for val in obj.predict(df_control)]

    global normalize_vars
    if normalize_vars is not None:
        pred_treat = denormalize(pd.DataFrame(pred_treat, columns=['y']), normalize_vars)['y']
        pred_control = denormalize(pd.DataFrame(pred_control, columns=['y']), normalize_vars)['y']

    pred_df = pd.DataFrame({
        "pr_y1_t1": pred_treat,
        "pr_y1_t0": pred_control,
    })
    return pred_df
