import pandas as pd
from utils.utils import normalize, denormalize
from deep.dnn import build_model, set_optimizer, init_seed

normalize_vars = None


def fit(x, y, t, **kwargs):
    lr = kwargs.get('lr')
    epochs = kwargs.get('epochs')
    batch_size = kwargs.get('batch_size')
    decay = kwargs.get('decay')
    method = kwargs.get('method')
    activation = 'sigmoid' if method == 'logistic' else 'linear'

    init_seed(1234)

    df = x.copy()
    df['treated'] = t

    global normalize_vars
    df, normalize_vars = normalize(df)

    model = build_model(df.shape[1], 1, activation)
    model = set_optimizer(model, lr, activation, decay)
    model.fit(df, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)

    return model


def predict(obj, newdata, **kwargs):
    df_treat = newdata.copy()
    df_control = newdata.copy()

    df_treat['treated'] = 1
    df_control['treated'] = 0

    global normalize_vars
    df_treat, _ = normalize(df_treat, normalize_vars)
    df_control, _ = normalize(df_control, normalize_vars)

    pred_treat = [val[0] for val in obj.predict(df_treat)]
    pred_control = [val[0] for val in obj.predict(df_control)]

    pred_df = pd.DataFrame({
        "pr_y1_t1": pred_treat,
        "pr_y1_t0": pred_control,
    })
    return pred_df
