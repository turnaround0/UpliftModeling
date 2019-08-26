import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression


def z_assign(y, t):
    """ Define transformed response variable z
    if (treated and response) or (not treated and not response), return 1
    else, return 0
    """
    if y == 1 and t == 1:
        return 1
    elif y == 0 and t == 1:
        return 0
    elif y == 1 and t == 0:
        return 0
    elif y == 0 and t == 0:
        return 1
    else:
        return None


def fit(x, y, t, method=LogisticRegression, **kwargs):
    """Transforming the data according to the "Jaskowski's Approach"
    Sometimes, it called Response Variable Transformation for Uplift (RVTU)

    Source: "Uplift modeling for clinical trial data" (Jaskowski, 2006)
    """

    ### Combine x, y, and ct
    df = x.copy()
    df['y'] = y
    df['ct'] = t
    df['z'] = df.apply(lambda row: z_assign(row['y'], row['ct']), axis=1)

    mdl = method(**kwargs).fit(x, df['z'])

    return mdl


def predict(obj, newdata, **kwargs):
    # df = pd.DataFrame({'y': y.copy()})
    # df['ct'] = ct
    # z = df.apply(lambda row: z_assign(row['y'], row['ct']), axis=1)

    if isinstance(obj, LinearRegression):
        pred = obj.predict(newdata)
    else:
        pred = obj.predict_proba(newdata)[:, 1]

    res = pd.DataFrame({
        "pr_y1_t1": [row for row in pred],
        "pr_y1_t0": [1 - row for row in pred],
    })
    return res
