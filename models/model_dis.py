import pandas as pd
from over.gan import get_stored_discriminator


def fit(x, y, t, **kwargs):
    return get_stored_discriminator()


def predict(obj, newdata, **kwargs):
    df = newdata.copy()
    discriminator = obj
    pred_list = []
    for t, y in [(1, 1), (1, 0), (0, 1), (0, 0)]:
        df['T'] = t
        df['Y'] = y
        discriminator.trainable = False
        pred = discriminator.predict(df)
        pred_list.append(pd.DataFrame([val[0] for val in pred]))

    pred_df = pd.concat(pred_list, axis=1)
    pred_df.columns = ['tr', 'tn', 'cr', 'cn']

    pred_treat = pred_df['tr'].div(pred_df['tr'] + pred_df['tn'] + 1e-8)
    pred_control = pred_df['cr'].div(pred_df['cr'] + pred_df['cn'] + 1e-8)

    pred_df = pd.DataFrame({
        "pr_y1_t1": pred_treat,
        "pr_y1_t0": pred_control,
    })
    return pred_df
