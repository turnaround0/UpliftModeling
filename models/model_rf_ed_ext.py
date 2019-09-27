import numpy as np
import pandas as pd
from models import model_dt_ed_ext


# Execute Random forest tree algorithm
# When building tree, decision tree extraction method is applied.
def set_params(max_round, p_value):
    model_dt_ed_ext.set_params(max_round, p_value)


# Random forest tree model
def fit(x, y, t, ntree=10, bagging_fraction=0.6, random_seed=1234, **kwargs):
    print('number of trees:', ntree)
    print('bagging_fraction:', bagging_fraction)

    predict_attr = kwargs.get('predict_attr', 'Y')
    treatment_attr = kwargs.get('treatment_attr', 'T')

    df = x
    df[predict_attr] = y
    df[treatment_attr] = t
    df = df.copy().reset_index(drop=True)

    kwargs['predict_attr'] = predict_attr
    kwargs['treatment_attr'] = treatment_attr

    np.random.seed(random_seed)
    random_seeds = [np.random.randint(10000) for _ in range(ntree)]
    trees = []
    for i in range(ntree):
        print('#Tree:', i + 1)
        bagged_df = df.sample(frac=bagging_fraction, random_state=random_seeds[i])
        kwargs.update({'random_seed': random_seeds[i]})

        bagged_x = bagged_df.drop([predict_attr, treatment_attr], axis=1)
        bagged_y = bagged_df[predict_attr]
        bagged_t = bagged_df[treatment_attr]
        trees.append(model_dt_ed_ext.fit(bagged_x, bagged_y, bagged_t, **kwargs))

    return trees


def predict(obj, newdata, **kwargs):
    pred_trees = []
    for idx, tree_obj in enumerate(obj):
        print('#Tree:', idx + 1)
        pred_trees.append(model_dt_ed_ext.predict(tree_obj, newdata, **kwargs))

    pred_trees_df = pd.concat(pred_trees, axis=1)
    pred_df = pd.DataFrame({
        "pr_y1_t1": pred_trees_df['pr_y1_t1'].mean(axis=1),
        "pr_y1_t0": pred_trees_df['pr_y1_t0'].mean(axis=1),
    })
    return pred_df
