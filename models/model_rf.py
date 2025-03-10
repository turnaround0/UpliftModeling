import numpy as np
import pandas as pd
from tree import tree, tree_bin


# Random forest tree model
def fit(x, y, t, ntree=10, bagging_fraction=0.6, random_seed=1234, **kwargs):
    predict_attr = kwargs.get('predict_attr', 'Y')
    treatment_attr = kwargs.get('treatment_attr', 'T')

    df = x.copy()
    df[predict_attr] = y
    df[treatment_attr] = t

    kwargs['predict_attr'] = predict_attr
    kwargs['treatment_attr'] = treatment_attr

    np.random.seed(random_seed)
    random_seeds = [np.random.randint(10000) for _ in range(ntree)]
    trees = []
    for i in range(ntree):
        bagged_df = df.sample(frac=bagging_fraction, random_state=random_seeds[i])
        if kwargs.get('bins') is None:
            trees.append(tree.build_tree(bagged_df, x.columns, random_seed=random_seeds[i], **kwargs))
        else:
            trees.append(tree_bin.build_tree(bagged_df, x.columns, random_seed=random_seeds[i], **kwargs))

    return trees


def predict(obj, newdata, **kwargs):
    pred_trees = []
    for tree_obj in obj:
        if kwargs.get('bins') is None:
            pred_trees.append(tree.test_predictions(tree_obj, newdata))
        else:
            pred_trees.append(tree_bin.test_predictions(tree_obj, newdata))

    pred_df = pd.DataFrame({
        "pr_y1_t1": sum([x['pr_y1_t1'] for x in pred_trees]) / len(pred_trees),
        "pr_y1_t0": sum([x['pr_y1_t0'] for x in pred_trees]) / len(pred_trees),
    })
    return pred_df
