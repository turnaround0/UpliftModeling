from tree import build_tree, test_predictions


# Tree model
def fit(x, y, t, **kwargs):
    predict_attr = kwargs.get('predict_attr', 'Y')
    treatment_attr = kwargs.get('treatment_attr', 'T')

    df = x.copy()
    df[predict_attr] = y
    df[treatment_attr] = t

    kwargs['predict_attr'] = predict_attr
    kwargs['treatment_attr'] = treatment_attr
    root = build_tree(df, x.columns, **kwargs)

    return root


def predict(root, newdata, **kwargs):
    return test_predictions(root, newdata)