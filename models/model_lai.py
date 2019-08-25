import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


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


def fit(x, y, t, method=GradientBoostingClassifier, **kwargs):
    """Training a model according to the "Lai's Approach"
    The default model is Gradient Boosting Machine (gbm)

    Source: "Influential Marketing" (Lai, 2006) and "Mining Truly Responsive
            Customers Using True Lift Overview" (Kane, 2014)

    Args:
        x: A data frame of predictors.
        y: A binary response (numeric) vector.
        t: A binary response (numeric) representing the treatment assignment
            (coded as 0/1).
        method: A sklearn model specifying which classification or regression
            model to use. This should be a method that can handle a
            multinominal class variable.

    Return:
        A sklearn model.
    """
    df = pd.DataFrame({'y': y.copy()})
    df['t'] = t
    z = df.apply(lambda row: z_assign(row['y'], row['t']), axis=1)

    model = method(**kwargs).fit(x, z)

    return model


def predict(obj, newdata, y, t, **kwargs):
    """Predictions according to the "Lai's Approach"

    Source: "Influential Marketing" (Lai, 2006) and "Mining Truly Responsive
            Customers Using True Lift Overview" (Kane, 2014)

    Args:
        obj: A sklearn model.
        newdata: A data frame containing the values at which predictions
            are required.

    Return:
        dataframe: A dataframe with predictions for when the instances are
            treated and for when they are not treated.
    """
    pred = obj.predict_proba(newdata)  # list of [False, True]

    res = pd.DataFrame({
        "pr_y1_t1": [row[1] for row in pred],
        "pr_y1_t0": [row[0] for row in pred],
    })
    return res
