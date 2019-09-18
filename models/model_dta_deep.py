import pandas as pd
import tensorflow as tf
from tensorflow import keras


def build_model_for_linear(x_len):
    model = keras.Sequential([
        keras.layers.Dense(256, input_shape=(x_len,), kernel_initializer='normal'),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(128, kernel_initializer='normal'),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(32, kernel_initializer='normal'),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(1, kernel_initializer='normal'),
    ])

    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
    # model.summary()

    return model


def build_model(x_len):
    model = keras.Sequential([
        keras.layers.Dense(128, input_shape=(x_len,)),
        keras.layers.LeakyReLU(),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(32),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(8),
        keras.layers.LeakyReLU(),

        keras.layers.Dense(1, activation='sigmoid'),
    ])

    adam = keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()

    return model


def fit(x, y, t, **kwargs):
    tf.compat.v1.set_random_seed(1234)

    df = x.copy()
    for col_name in x.columns:
        df["Int_" + col_name] = x[col_name] * t
    df['treated'] = t

    if kwargs.get('method') == 'logistic':
        model = build_model(df.shape[1])
    else:
        model = build_model_for_linear(df.shape[1])

    model.fit(df, y, epochs=50, batch_size=64)

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
