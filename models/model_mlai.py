import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from dataset.preprocess import ty_assign
from utils.utils import num_class

ext_params = {}


def set_params(class_weight):
    global ext_params

    ext_params = {
        'class_weight': class_weight,
    }
    print('Extraction params:', ext_params)


def find_best_mlai_params(x, y, t):
    print('Start finding best MLAI params')

    df = x.copy()
    df['Y'] = y
    df['T'] = t

    tr, tn, cr, cn = num_class(df, 'Y', 'T')
    pr_y1_t1 = tr / (tr + tn)
    pr_y1_t0 = cr / (cr + cn)
    pr_y0_t1 = tn / (tr + tn)
    pr_y0_t0 = cn / (cr + cn)

    # similarity for cn with tr
    ed_gain = (pr_y1_t1 - pr_y0_t0) ** 2
    gini_tr_cn = 2 * pr_y1_t1 * pr_y0_t0 * (1 - pr_y1_t1) * (1 - pr_y0_t0)
    gini_tr = 2 * pr_y1_t1 * (1 - pr_y1_t1)
    gini_cn = 2 * pr_y0_t0 * (1 - pr_y0_t0)
    ed_norm = gini_tr_cn * ed_gain + gini_tr * pr_y1_t1 + gini_cn * pr_y0_t0 + 0.5
    tr_cn = ed_gain / ed_norm

    # similarity for tn with cr
    ed_gain = (pr_y0_t1 - pr_y1_t0) ** 2
    gini_tn_cr = 2 * pr_y0_t1 * pr_y1_t0 * (1 - pr_y0_t1) * (1 - pr_y1_t0)
    gini_tn = 2 * pr_y0_t1 * (1 - pr_y0_t1)
    gini_cr = 2 * pr_y1_t0 * (1 - pr_y1_t0)
    ed_norm = gini_tn_cr * ed_gain + gini_tn * pr_y0_t1 + gini_cr * pr_y1_t0 + 0.5
    cr_tn = ed_gain / ed_norm

    print('Best MLAI params:', tr_cn, cr_tn)
    return tr_cn, cr_tn


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
    if ext_params['class_weight'] == 'calculate':
        alpha, beta = find_best_mlai_params(x, y, t)
        kwargs.update({'class_weight': {'CN': alpha, 'CR': 1, 'TN': beta, 'TR': 1}})
    else:
        kwargs.update({'class_weight': ext_params['class_weight']})

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
    t = kwargs.get('t')
    prob_T = sum(t) / len(t)
    prob_C = 1 - prob_T

    pred = obj.predict_proba(newdata)  # list of [CN, CR, TN, TR]

    res = pd.DataFrame({
        "pr_y1_t1": [row[3] / prob_T + row[0] / prob_C for row in pred],  # TR/T + CN/C
        "pr_y1_t0": [row[2] / prob_T + row[1] / prob_C for row in pred],  # TN/T + CR/C
    })
    return res
