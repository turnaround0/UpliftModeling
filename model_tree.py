import sys
import numpy as np
import random
import math

import warnings

warnings.filterwarnings("ignore")


class Node(object):
    def __init__(self, attribute, threshold):
        self.attr = attribute
        self.thres = threshold
        self.left = None
        self.right = None
        self.leaf = False
        self.predict = None, None


def info_gain(df, attribute, predict_attr, treatment_attr,
              method, min_bucket_t0, min_bucket_t1):
    """
    Select the information gain and threshold of the attribute to split
    The threshold chosen splits the test data such that information gain is maximized

    Return a pandas.DataFrame
        columns: 'thres' (threshold) and 'info_gain' (information gain)
    """
    num_total = df.shape[0]
    tmp = pd.DataFrame({
        'thres': df[attribute],
        'Y': df[predict_attr],
        'T': df[treatment_attr]
    })
    tmp.sort_values(['thres'], inplace=True)

    tmp['n_t1_L'] = (tmp['T']).cumsum()
    tmp['n_t0_L'] = (tmp['T'] == 0).cumsum()
    tmp['n_t1_R'] = sum(tmp['T']) - (tmp['T']).cumsum()
    tmp['n_t0_R'] = sum(tmp['T'] == 0) - (tmp['T'] == 0).cumsum()
    tmp['n_y_t1_L'] = (tmp['T'] & tmp['Y']).cumsum()
    tmp['n_y_t0_L'] = ((tmp['T'] == 0) & tmp['Y']).cumsum()
    tmp['n_y_t1_R'] = sum(tmp['T'] & tmp['Y']) - (tmp['T'] & tmp['Y']).cumsum()
    tmp['n_y_t0_R'] = sum((tmp['T'] == 0) & tmp['Y']) - ((tmp['T'] == 0) & tmp['Y']).cumsum()

    # min bucket condition
    #   Check the size of treatment & control group in left & right child
    tmp['min_bucket_ok'] = ((tmp['n_t1_L'] >= min_bucket_t1) &
                            (tmp['n_t0_L'] >= min_bucket_t0) &
                            (tmp['n_t1_R'] >= min_bucket_t1) &
                            (tmp['n_t0_R'] >= min_bucket_t0))

    if sum(tmp['min_bucket_ok']) > 0:
        num_total = df.shape[0]
        tr, tn, cr, cn = num_class(df, predict_attr, treatment_attr)
        n_t1 = tr + tn
        n_t0 = cr + cn
        pr_t1 = (tr + tn) / (num_total)
        # r_t0 = (tr + cr) / (num_total)
        pr_t0 = 1 - pr_t1
        pr_y1_t1 = tr / (tr + tn)
        pr_y1_t0 = cr / (cr + cn)

        # Randomized assignment implies pr_l_t1 = pr_l_t0 for all possible splits
        pr_l_t1 = (tmp['n_t1_L']) / (n_t1)
        pr_l_t0 = (tmp['n_t0_L']) / (n_t0)
        pr_l = pr_l_t1 * pr_t1 + pr_l_t0 * pr_t0
        pr_r = 1 - pr_l

        # Add Laplace correction to probablities
        pr_y1_l_t1 = (tmp['n_y_t1_L']) / (tmp['n_t1_L'])
        pr_y1_l_t0 = (tmp['n_y_t0_L']) / (tmp['n_t0_L'])
        pr_y1_r_t1 = (tmp['n_y_t1_R']) / (tmp['n_t1_R'])
        pr_y1_r_t0 = (tmp['n_y_t0_R']) / (tmp['n_t0_R'])

        # Number of treatment/control observations at left and right child nodes
        n_t1_L = tmp['n_t1_L']
        n_t0_L = tmp['n_t0_L']
        n_t1_R = tmp['n_t1_R']
        n_t0_R = tmp['n_t0_R']

        if method.lower() == 'ed':
            tmp['info_gain'] = eucli_dist(tmp,
                                          pr_y1_t1,
                                          pr_y1_t0,
                                          pr_l,
                                          pr_r,
                                          pr_y1_l_t1,
                                          pr_y1_l_t0,
                                          pr_y1_r_t1,
                                          pr_y1_r_t0,
                                          pr_t1,
                                          pr_t0,
                                          pr_l_t1,
                                          pr_l_t0)
        elif method.lower() == 'kl':
            tmp['info_gain'] = kl_divergence(tmp,
                                             pr_y1_t1,
                                             pr_y1_t0,
                                             pr_l,
                                             pr_r,
                                             pr_y1_l_t1,
                                             pr_y1_l_t0,
                                             pr_y1_r_t1,
                                             pr_y1_r_t0,
                                             pr_t1,
                                             pr_t0,
                                             pr_l_t1,
                                             pr_l_t0)
        elif method.lower() == 'chisq':
            tmp['info_gain'] = chisq(tmp,
                                     pr_y1_t1,
                                     pr_y1_t0,
                                     pr_l,
                                     pr_r,
                                     pr_y1_l_t1,
                                     pr_y1_l_t0,
                                     pr_y1_r_t1,
                                     pr_y1_r_t0,
                                     pr_t1,
                                     pr_t0,
                                     pr_l_t1,
                                     pr_l_t0)
        elif method.lower() == 'int':
            tmp['info_gain'] = interaction_split(tmp,
                                                 pr_y1_t1,
                                                 pr_y1_t0,
                                                 pr_l,
                                                 pr_r,
                                                 pr_y1_l_t1,
                                                 pr_y1_l_t0,
                                                 pr_y1_r_t1,
                                                 pr_y1_r_t0,
                                                 pr_t1,
                                                 pr_t0,
                                                 pr_l_t1,
                                                 pr_l_t0,
                                                 n_t1_L,
                                                 n_t0_L,
                                                 n_t1_R,
                                                 n_t0_R)
        else:
            raise NotImplementedError

    # We will select one rows per one distinct candidate
    tmp['dups'] = tmp['thres'].duplicated(keep='last')
    tmp['thres_ok'] = (tmp['min_bucket_ok'] & (tmp['dups'] == False))
    tmp.dropna(inplace=True)
    if sum(tmp['thres_ok']) < 1:
        return None

    tmp = tmp[tmp['thres_ok']]

    return tmp[['thres', 'info_gain']]


def num_class(df, predict_attr, treatment_attr):
    """
    Returns the number of Responders and Non-responders in Treatment and Control group
    """
    tr = df[(df[predict_attr] == 1) & (df[treatment_attr] == 1)]  # Responders in Treatment group
    tn = df[(df[predict_attr] == 0) & (df[treatment_attr] == 1)]  # Non-responders in Treatment group
    cr = df[(df[predict_attr] == 1) & (df[treatment_attr] == 0)]  # Responders in Control group
    cn = df[(df[predict_attr] == 0) & (df[treatment_attr] == 0)]  # Non-responders in Control group
    return tr.shape[0], tn.shape[0], cr.shape[0], cn.shape[0]


def choose_attr(df, attributes, predict_attr, treatment_attr,
                method, min_bucket_t0, min_bucket_t1):
    """
    Chooses the attribute and its threshold with the highest info gain
    from the set of attributes
    """
    max_info_gain = 0
    best_attr = None
    threshold = None
    # Test each attribute (note attributes maybe be chosen more than once)
    for attr in attributes:
        df_ig = info_gain(df, attr, predict_attr, treatment_attr,
                          method, min_bucket_t0, min_bucket_t1)
        if df_ig is None:
            continue

        # Get the possible indices of maximum info gain
        ig = max(df_ig['info_gain'])
        idx_ig = df_ig.index[df_ig['info_gain'] == ig]
        # Break ties randomly
        idx_ig = random.choice(idx_ig)
        # Get information gain & threshold of that
        thres = df_ig['thres'][idx_ig]

        if ig > max_info_gain:
            max_info_gain = ig
            best_attr = attr
            threshold = thres
    return best_attr, threshold


def build_tree(df, cols, predict_attr='Y', treatment_attr='T',
               method='ED', depth=1, max_depth=float('INF'),
               min_split=2000, min_bucket_t0=None, min_bucket_t1=None,
               mtry=None, random_seed=1234):
    """
    Builds the Decision Tree based on training data, attributes to train on,
    and a prediction attribute
    """
    if depth == 1:
        np.random.seed(random_seed)

    if mtry is None:
        mtry = math.floor(math.sqrt(len(cols)))
    if min_bucket_t0 is None:
        min_bucket_t0 = round(min_split / 4)
    if min_bucket_t1 is None:
        min_bucket_t1 = round(min_split / 4)

    # Get the number of positive and negative examples in the training data
    tr, tn, cr, cn = num_class(df, predict_attr, treatment_attr)
    r_y1_ct1 = tr / (tr + tn)
    r_y1_ct0 = cr / (cr + cn)

    # Check variables have less than 2 levels at the current node
    # If not, exclude them as candidates for mtry selection
    # To split the node, sum(ok_vars) should be equal or larger than self.mtry
    ok_vars = []
    for col in cols:
        ok_vars.append(len(set(df[col])) > 1)

    # Whether we have to split this node
    #   1. min split condition: Both the sizes of treatment and control group
    #     of an internal node should be larger than 'min_split'
    #   2. max depth condition: The depth of tree is 'max_depth'
    #   3. min_bucket condition: The number of treatment/control group of a
    #     node should be larger than 'min_bucket_t0'/'min_bucket_t1'
    #   4. Expected return should be larger than 0 and smaller than 1
    #     (for KL-divergence & Chisq splitting criteria)
    split_cond = tr + tn > min_split and cr + cn > min_split \
                 and 0 < r_y1_ct1 < 1 and 0 < r_y1_ct0 < 1 \
                 and depth < max_depth and sum(ok_vars) >= mtry

    best_attr, threshold = None, None
    if split_cond:
        # Sample columns
        ok_cols = [col for col in cols if len(set(df[col])) > 1]
        ok_cols = np.random.choice(ok_cols, mtry, replace=False)
        # Determine attribute and its threshold value with the highest
        # information gain
        best_attr, threshold = choose_attr(df, ok_cols, predict_attr, treatment_attr,
                                           method, min_bucket_t0, min_bucket_t1)
    if best_attr is None:
        # Create a leaf node indicating it's prediction
        leaf = Node(None, None)
        leaf.leaf = True
        leaf.predict = (tr / (tr + tn), cr / (cr + cn))
        return leaf
    else:
        # Create internal tree node based on attribute and it's threshold
        sub_1 = df[df[best_attr] <= threshold]
        sub_2 = df[df[best_attr] > threshold]
        sub1_tr, sub1_tn, sub1_cr, sub1_cn = num_class(sub_1, predict_attr, treatment_attr)
        sub2_tr, sub2_tn, sub2_cr, sub2_cn = num_class(sub_2, predict_attr, treatment_attr)
        tree = Node(best_attr, threshold)
        # Recursively build left and right subtree
        tree.left = build_tree(sub_1, cols, predict_attr, treatment_attr,
                               method=method, depth=depth + 1, max_depth=max_depth,
                               min_split=min_split, min_bucket_t0=min_bucket_t0,
                               min_bucket_t1=min_bucket_t1, mtry=mtry)
        tree.right = build_tree(sub_2, cols, predict_attr, treatment_attr,
                                method=method, depth=depth + 1, max_depth=max_depth,
                                min_split=min_split, min_bucket_t0=min_bucket_t0,
                                min_bucket_t1=min_bucket_t1, mtry=mtry)
        return tree


def predict(node, row_df):
    """
    Given a instance of a training data, make a prediction of an observation (row)
    based on the Decision Tree
    Assumes all data has been cleaned (i.e. no NULL data)
    """
    # If we are at a leaf node, return the prediction of the leaf node
    if node.leaf:
        return node.predict
    # Traverse left or right subtree based on instance's data
    if row_df[node.attr] <= node.thres:
        return predict(node.left, row_df)
    elif row_df[node.attr] > node.thres:
        return predict(node.right, row_df)


def test_predictions(root, df):
    """
    Given a set of data, make a prediction for each instance using the Decision Tree
    """
    pred_treat = []
    pred_control = []
    for index, row in df.iterrows():
        return_treated, return_control = predict(root, row)
        pred_treat.append(return_treated)
        pred_control.append(return_control)
    pred_df = pd.DataFrame({
        "pr_y1_t1": pred_treat,
        "pr_y1_t0": pred_control,
    })
    return pred_df


def eucli_dist(small_df,
               pr_y1_ct1,
               pr_y1_ct0,
               pr_l,
               pr_r,
               pr_y1_l_ct1,
               pr_y1_l_ct0,
               pr_y1_r_ct1,
               pr_y1_r_ct0,
               pr_ct1,
               pr_ct0,
               pr_l_ct1,
               pr_l_ct0):
    # Euclidean gain
    ed_node = (pr_y1_ct1 - pr_y1_ct0) ** 2 + ((1 - pr_y1_ct1) - (1 - pr_y1_ct0)) ** 2
    ed_l = (pr_y1_l_ct1 - pr_y1_l_ct0) ** 2 + ((1 - pr_y1_l_ct1) - (1 - pr_y1_l_ct0)) ** 2
    ed_r = (pr_y1_r_ct1 - pr_y1_r_ct0) ** 2 + ((1 - pr_y1_r_ct1) - (1 - pr_y1_r_ct0)) ** 2
    ed_lr = pr_l * ed_l + pr_r * ed_r
    ed_gain = ed_lr - ed_node

    # Euclidean Normalization factor
    gini_ct = 2 * pr_ct1 * (1 - pr_ct1)
    ed_ct = (pr_l_ct1 - pr_l_ct0) ** 2 + ((1 - pr_l_ct1) - (1 - pr_l_ct0)) ** 2
    gini_ct1 = 2 * pr_l_ct1 * (1 - pr_l_ct1)
    gini_ct0 = 2 * pr_l_ct0 * (1 - pr_l_ct0)
    ed_norm = gini_ct * ed_ct + gini_ct1 * pr_ct1 + gini_ct0 * pr_ct0 + 0.5

    # Output
    info_gain_t = ed_gain / ed_norm

    return info_gain_t


def kl_divergence(small_df,
                  pr_y1_ct1,
                  pr_y1_ct0,
                  pr_l,
                  pr_r,
                  pr_y1_l_ct1,
                  pr_y1_l_ct0,
                  pr_y1_r_ct1,
                  pr_y1_r_ct0,
                  pr_ct1,
                  pr_ct0,
                  pr_l_ct1,
                  pr_l_ct0):
    # KL Gain
    kl_node = pr_y1_ct1 * np.log2(pr_y1_ct1 / pr_y1_ct0) + \
              (1 - pr_y1_ct1) * np.log2((1 - pr_y1_ct1) / (1 - pr_y1_ct0))
    kl_l = pr_y1_l_ct1 * np.log2(pr_y1_l_ct1 / pr_y1_l_ct0) + \
           (1 - pr_y1_l_ct1) * np.log2((1 - pr_y1_l_ct1) / (1 - pr_y1_l_ct0))
    kl_r = pr_y1_r_ct1 * np.log2(pr_y1_r_ct1 / pr_y1_r_ct0) + \
           (1 - pr_y1_r_ct1) * np.log2((1 - pr_y1_r_ct1) / (1 - pr_y1_r_ct0))
    kl_lr = pr_l * kl_l + pr_r * kl_r
    kl_gain = kl_lr - kl_node

    # KL Normalization factor
    ent_ct = -(pr_ct1 * np.log2(pr_ct1) + pr_ct0 * np.log2(pr_ct0))
    kl_ct = pr_l_ct1 * np.log2(pr_l_ct1 / pr_l_ct0) + \
            (1 - pr_l_ct1) * np.log2((1 - pr_l_ct1) / (1 - pr_l_ct0))
    ent_ct1 = -(pr_l_ct1 * np.log2(pr_l_ct1) + (1 - pr_l_ct1) * np.log2((1 - pr_l_ct1)))
    ent_ct0 = -(pr_l_ct0 * np.log2(pr_l_ct0) + (1 - pr_l_ct0) * np.log2((1 - pr_l_ct0)))

    norm = kl_ct * ent_ct + ent_ct1 * pr_ct1 + ent_ct0 * pr_ct0 + 0.5

    # Output
    info_gain_t = kl_gain / norm

    return info_gain_t


def chisq(small_df,
          pr_y1_ct1,
          pr_y1_ct0,
          pr_l,
          pr_r,
          pr_y1_l_ct1,
          pr_y1_l_ct0,
          pr_y1_r_ct1,
          pr_y1_r_ct0,
          pr_ct1,
          pr_ct0,
          pr_l_ct1,
          pr_l_ct0):
    # Chi-squared gain
    chisq_node = ((pr_y1_ct1 - pr_y1_ct0) ** 2) / pr_y1_ct0 + \
                 (((1 - pr_y1_ct1) - (1 - pr_y1_ct0)) ** 2) / (1 - pr_y1_ct0)
    chisq_l = ((pr_y1_l_ct1 - pr_y1_l_ct0) ** 2) / pr_y1_l_ct0 + \
              (((1 - pr_y1_l_ct1) - (1 - pr_y1_l_ct0)) ** 2) / (1 - pr_y1_l_ct0)
    chisq_r = ((pr_y1_r_ct1 - pr_y1_r_ct0) ** 2) / pr_y1_r_ct0 + \
              (((1 - pr_y1_r_ct1) - (1 - pr_y1_r_ct0)) ** 2) / (1 - pr_y1_r_ct0)
    chisq_lr = pr_l * chisq_l + pr_r * chisq_r
    chisq_gain = chisq_lr - chisq_node

    # Chi-squared Normalization factor
    gini_ct = 2 * pr_ct1 * (1 - pr_ct1)
    chisq_ct = ((pr_l_ct1 - pr_l_ct0) ** 2) / pr_l_ct0 + \
               (((1 - pr_l_ct1) - (1 - pr_l_ct0)) ** 2) / (1 - pr_l_ct0)
    gini_ct1 = 2 * pr_l_ct1 * (1 - pr_l_ct1)
    gini_ct0 = 2 * pr_l_ct0 * (1 - pr_l_ct0)
    chisq_norm = gini_ct * chisq_ct + gini_ct1 * pr_ct1 + gini_ct0 * pr_ct0 + 0.5

    # Output
    info_gain_t = chisq_gain / chisq_norm

    return info_gain_t


def interaction_split(small_df,
                      pr_y1_ct1,
                      pr_y1_ct0,
                      pr_l,
                      pr_r,
                      pr_y1_l_ct1,
                      pr_y1_l_ct0,
                      pr_y1_r_ct1,
                      pr_y1_r_ct0,
                      pr_ct1,
                      pr_ct0,
                      pr_l_ct1,
                      pr_l_ct0,
                      cs_ct1,
                      cs_ct0,
                      ncs_ct1,
                      ncs_ct0):
    # Compute elements for split formula
    C44 = 1 / cs_ct1 + 1 / cs_ct0 + 1 / ncs_ct1 + 1 / ncs_ct0

    UR = pr_y1_r_ct1 - pr_y1_r_ct0
    UL = pr_y1_l_ct1 - pr_y1_l_ct0

    SSE = cs_ct1 * pr_y1_l_ct1 * (1 - pr_y1_l_ct1) + \
          ncs_ct1 * pr_y1_r_ct1 * (1 - pr_y1_r_ct1) + \
          cs_ct0 * pr_y1_l_ct0 * (1 - pr_y1_l_ct0) + \
          ncs_ct0 * pr_y1_r_ct0 * (1 - pr_y1_r_ct0)

    n_node = len(small_df)

    # Output: Interaction split
    info_gain_t = ((n_node - 4) * (UR - UL) ** 2) / (C44 * SSE)

    return info_gain_t


def uplift_tree(x, y, t, **kwargs):
    predict_attr = kwargs.get('predict_attr', 'Y')
    treatment_attr = kwargs.get('treatment_attr', 'T')

    df = x.copy()
    df[predict_attr] = y
    df[treatment_attr] = t

    kwargs['predict_attr'] = predict_attr
    kwargs['treatment_attr'] = treatment_attr
    root = build_tree(df, x.columns, **kwargs)

    return root


def predict_tree(root, newdata, **kwargs):
    return test_predictions(root, newdata)


def upliftRF(x, y, t, ntree=10, bagging_fraction=0.6, random_seed=1234, **kwargs):
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
        trees.append(build_tree(bagged_df, x.columns, random_seed=random_seeds[i], **kwargs))

    return trees


def predict_upliftRF(obj, newdata, **kwargs):
    pred_trees = []
    for tree in obj:
        pred_trees.append(test_predictions(tree, newdata))

    pred_df = pd.DataFrame({
        "pr_y1_t1": sum([x['pr_y1_t1'] for x in pred_trees]) / len(pred_trees),
        "pr_y1_t0": sum([x['pr_y1_t0'] for x in pred_trees]) / len(pred_trees),
    })
    return pred_df
