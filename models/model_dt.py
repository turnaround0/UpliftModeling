from tree import tree, tree_bin


# Decision tree model
def fit(x, y, t, **kwargs):
    predict_attr = kwargs.get('predict_attr', 'Y')
    treatment_attr = kwargs.get('treatment_attr', 'T')

    df = x.copy()
    df[predict_attr] = y
    df[treatment_attr] = t

    kwargs['predict_attr'] = predict_attr
    kwargs['treatment_attr'] = treatment_attr

    if kwargs.get('bins') is None:
        return tree.build_tree(df, x.columns, **kwargs)
    else:
        return tree_bin.build_tree(df, x.columns, **kwargs)


def predict(obj, newdata, **kwargs):
    return tree.test_predictions(obj, newdata)
