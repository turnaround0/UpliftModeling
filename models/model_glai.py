import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from preprocess import ty_assign


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
    ty = df.apply(lambda row: ty_assign(row['y'], row['t']), axis=1)

    model = method(**kwargs).fit(x, ty)

    return model


def predict(obj, newdata, **kwargs):
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
    t = kwargs.get('T')
    prob_T = sum(t) / len(t)
    prob_C = 1 - prob_T

    pred = obj.predict_proba(newdata)  # list of [CN, CR, TN, TR]

    res = pd.DataFrame({
        "pr_y1_t1": [row[3] / prob_T + row[0] / prob_C for row in pred],  # TR/T + CN/C
        "pr_y1_t0": [row[2] / prob_T + row[1] / prob_C for row in pred],  # TN/T + CR/C
    })
    return res
