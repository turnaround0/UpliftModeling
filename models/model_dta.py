from sklearn.linear_model import LogisticRegression
import pandas as pd


def fit(x, y, t, method=LogisticRegression, **kwargs):
    """Training a model according to the "Dummy Treatment Approach" 
    The default model is General Linear Model (GLM)

    Source: "The True Lift Model" (Lo, 2002)

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
    # Create interaction variables
    # Building our dataframe with the interaction variables
    df = x.copy()
    for colname in x.columns:
        df["Int_" + colname] = x[colname] * t
    df['treated'] = t

    # Fit a model
    model = method(**kwargs).fit(df, y)

    return model


def predict(obj, newdata, y_name='y', t_name='treated', **kwargs):
    """Predictions according to the "Dummy Treatment Approach" 

    For each instance in newdata two predictions are made:
    1) What is the probability of a person responding when treated?
    2) What is the probability of a person responding when not treated
      (i.e. part of control group)?

    Source: "The True Lift Model" (Lo, 2002)

    Args:
        obj: A sklearn model.
        newdata: A data frame containing the values at which predictions
            are required.

    Return:
        dataframe: A dataframe with predictions for when the instances are
            treated and for when they are not treated.
    """
    predictors = [c for c in newdata.columns if c not in (y_name, t_name)]

    df_treat = newdata.copy()
    df_control = newdata.copy()
    for colname in predictors:
        df_treat["Int_" + colname] = df_treat[colname] * 1
        df_control["Int_" + colname] = df_control[colname] * 0
    df_treat['treated'] = 1
    df_control['treated'] = 0

    pred_treat = obj.predict_proba(df_treat)[:, 1]
    pred_control = obj.predict_proba(df_control)[:, 1]

    pred_df = pd.DataFrame({
        "pr_y1_t1": pred_treat,
        "pr_y1_t0": pred_control,
    })
    return pred_df
