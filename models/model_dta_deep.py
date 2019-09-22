import pandas as pd
import tensorflow as tf
from tensorflow import keras


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
    for col_name in x.columns:
        df["Int_" + col_name] = x[col_name] * t
    df['treated'] = t

    model = build_model(df.shape[1], lr, activation)
    model.fit(df, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    return model


def predict(obj, newdata, y_name='y', t_name='treated', **kwargs):
    predictors = [c for c in newdata.columns if c not in (y_name, t_name)]

    df_treat = newdata.copy()
    df_control = newdata.copy()

    for col_name in predictors:
        df_treat["Int_" + col_name] = df_treat[col_name] * 1
        df_control["Int_" + col_name] = df_control[col_name] * 0
    df_treat['treated'] = 1
    df_control['treated'] = 0

    pred_treat = [val[0] for val in obj.predict(df_treat)]
    pred_control = [val[0] for val in obj.predict(df_control)]

    pred_df = pd.DataFrame({
        "pr_y1_t1": pred_treat,
        "pr_y1_t0": pred_control,
    })
    return pred_df
