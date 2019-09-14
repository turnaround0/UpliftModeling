import pandas as pd

repeat_num = 10


def repeat(df):
    return pd.concat([df] * repeat_num, axis=0).reset_index(drop=True)


def over_sampling(X, T, Y):
    return repeat(X), repeat(T), repeat(Y)
