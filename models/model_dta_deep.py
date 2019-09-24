import pandas as pd
from deep.dnn import build_model, set_optimizer, init_seed


def fit(x, y, t, **kwargs):
    lr = kwargs.get('lr')
    epochs = kwargs.get('epochs')
    batch_size = kwargs.get('batch_size')
    decay = kwargs.get('decay')
    method = kwargs.get('method')
    activation = 'sigmoid' if method == 'logistic' else 'linear'

    init_seed(1234)

    df = x.copy()
    for col_name in x.columns:
        df["Int_" + col_name] = x[col_name] * t
    df['treated'] = t

    model = build_model(df.shape[1], 1, activation)
    model = set_optimizer(model, lr, activation, decay)
    model.fit(df, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)

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
