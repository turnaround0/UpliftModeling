import numpy as np
import pandas as pd
import random
import math

from tree.split_criterion_bin import eucli_dist, kl_divergence, chisq

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
              method, min_bucket_t0, min_bucket_t1, bins):
    """
    Select the information gain and threshold of the attribute to split
    The threshold chosen splits the test data such that information gain is maximized

    Return a pandas.DataFrame
        columns: 'thres' (threshold) and 'info_gain' (information gain)
    """
    num_total = df.shape[0]
    nbins = len(bins) - 1
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

    # Ratio of treatment/control group with Laplace correction (Laplace smoothing)
    pr_t1 = (tmp['T'].sum() + 1) / (num_total + 2)
    pr_t0 = 1 - pr_t1

    # Counts of data for each bin
    tmp['bin'] = pd.cut(tmp['Y'], bins=bins, labels=range(1, len(bins)))
    y_t1 = tmp[tmp['T'] == 1].groupby('bin').count()['thres']
    y_t0 = tmp[tmp['T'] == 0].groupby('bin').count()['thres']
    y_t1_L = pd.DataFrame({bin_idx: ((tmp['T'] == 1) & (tmp['bin'] == bin_idx)).cumsum()
                           for bin_idx in range(1, len(bins))})
    y_t0_L = pd.DataFrame({bin_idx: ((tmp['T'] == 0) & (tmp['bin'] == bin_idx)).cumsum()
                           for bin_idx in range(1, len(bins))})
    tmp['1'] = 1
    pr_t1_L = tmp['1'].cumsum() / num_total
    pr_t0_L = 1 - pr_t1_L

    # Add Laplace correction (Laplace smoothing) before make probability distributions
    pr_y_t1 = (y_t1 + 1) / (sum(y_t1) + nbins)
    pr_y_t0 = (y_t0 + 1) / (sum(y_t0) + nbins)
    pr_y_t1_L = pd.DataFrame({bin_idx: (y_t1_L[bin_idx] + 1) / (tmp['n_t1_L'] + nbins)
                              for bin_idx in range(1, len(bins))})
    pr_y_t0_L = pd.DataFrame({bin_idx: (y_t0_L[bin_idx] + 1) / (tmp['n_t0_L'] + nbins)
                              for bin_idx in range(1, len(bins))})
    pr_y_t1_R = pd.DataFrame({bin_idx: (y_t1[bin_idx] - y_t1_L[bin_idx] + 1) / (tmp['n_t1_R'] + nbins)
                              for bin_idx in range(1, len(bins))})
    pr_y_t0_R = pd.DataFrame({bin_idx: (y_t0[bin_idx] - y_t0_L[bin_idx] + 1) / (tmp['n_t0_R'] + nbins)
                              for bin_idx in range(1, len(bins))})

    # min bucket condition
    #   Check the size of treatment & control group in left & right child
    tmp['min_bucket_ok'] = ((tmp['n_t1_L'] >= min_bucket_t1) &
                            (tmp['n_t0_L'] >= min_bucket_t0) &
                            (tmp['n_t1_R'] >= min_bucket_t1) &
                            (tmp['n_t0_R'] >= min_bucket_t0))

    if sum(tmp['min_bucket_ok']) > 0:
        # Treatment ratio with Laplace correction
        pr_y_l = (y_t1_L + y_t0_L + 1) / (y_t1 + y_t0 + 2)
        pr_y = (tmp.groupby('bin').count()['thres'] + 1) / (num_total + nbins)
        pr_l = (pr_y_l * pr_y).sum(axis=1)
        pr_r = 1 - pr_l

        if method.lower() == 'ed':
            tmp['info_gain'] = eucli_dist(tmp,
                                          pr_y_t1,
                                          pr_y_t0,
                                          pr_l,
                                          pr_r,
                                          pr_y_t1_L,
                                          pr_y_t0_L,
                                          pr_y_t1_R,
                                          pr_y_t0_R,
                                          pr_t1,
                                          pr_t0,
                                          pr_t1_L,
                                          pr_t0_L)
        elif method.lower() == 'kl':
            tmp['info_gain'] = kl_divergence(tmp,
                                             pr_y_t1,
                                             pr_y_t0,
                                             pr_l,
                                             pr_r,
                                             pr_y_t1_L,
                                             pr_y_t0_L,
                                             pr_y_t1_R,
                                             pr_y_t0_R,
                                             pr_t1,
                                             pr_t0,
                                             pr_t1_L,
                                             pr_t0_L)
        elif method.lower() == 'chisq':
            tmp['info_gain'] = chisq(tmp,
                                     pr_y_t1,
                                     pr_y_t0,
                                     pr_l,
                                     pr_r,
                                     pr_y_t1_L,
                                     pr_y_t0_L,
                                     pr_y_t1_R,
                                     pr_y_t0_R,
                                     pr_t1,
                                     pr_t0,
                                     pr_t1_L,
                                     pr_t0_L)
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
    tr = df[(df[predict_attr] != 0) & (df[treatment_attr] == 1)]  # Responders in Treatment group
    tn = df[(df[predict_attr] == 0) & (df[treatment_attr] == 1)]  # Non-responders in Treatment group
    cr = df[(df[predict_attr] != 0) & (df[treatment_attr] == 0)]  # Responders in Control group
    cn = df[(df[predict_attr] == 0) & (df[treatment_attr] == 0)]  # Non-responders in Control group
    return tr.shape[0], tn.shape[0], cr.shape[0], cn.shape[0]


def choose_attr(df, attributes, predict_attr, treatment_attr,
                method, min_bucket_t0, min_bucket_t1, bins):
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
                          method, min_bucket_t0, min_bucket_t1, bins)
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
               mtry=None, random_seed=1234, bins=None, nbin=10, **kwargs):
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
    if bins is None:
        if set(df[predict_attr]) != {0, 1}:
            min_y = min(df[predict_attr])
            max_y = max(df[predict_attr])
            bin_size = (max_y - min_y) / nbin
            bins = np.arange(min_y, max_y + bin_size / 2, bin_size)
            bins[0] = -float('inf')
            bins[-1] = float('inf')
        else:
            bins = np.array([-float('inf'), 0.5, float('inf')])

    # Get the number of positive and negative examples in the training data
    tr, tn, cr, cn = num_class(df, predict_attr, treatment_attr)

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
    #             and 0 < r_y1_ct1 < 1 and 0 < r_y1_ct0 < 1 \
    split_cond = tr + tn > min_split and cr + cn > min_split \
                 and depth < max_depth and sum(ok_vars) >= mtry

    best_attr, threshold = None, None
    if split_cond:
        # Sample columns
        ok_cols = [col for col in cols if len(set(df[col])) > 1]
        ok_cols = np.random.choice(ok_cols, mtry, replace=False)
        # Determine attribute and its threshold value with the highest
        # information gain
        best_attr, threshold = choose_attr(df, ok_cols, predict_attr, treatment_attr,
                                           method, min_bucket_t0, min_bucket_t1,
                                           bins=bins)
    if best_attr is None:
        # Create a leaf node indicating it's prediction
        leaf = Node(None, None)
        leaf.leaf = True
        leaf.predict = (sum(df[(df[treatment_attr] == 1)][predict_attr]) / (tr + tn),
                        sum(df[(df[treatment_attr] == 0)][predict_attr]) / (cr + cn))

        # Tree extraction method
        u_value = kwargs.get('u_value')
        if u_value is not None:
            ext_idx_list = kwargs['ext_idx_list']
            uplift = leaf.predict[0] - leaf.predict[1]
            if uplift > u_value:
                ext_idx_list += df.index.tolist()

        return leaf
    else:
        # Create internal tree node based on attribute and it's threshold
        sub_1 = df[df[best_attr] <= threshold]
        sub_2 = df[df[best_attr] > threshold]
        tree = Node(best_attr, threshold)
        # Recursively build left and right subtree
        tree.left = build_tree(sub_1, cols, predict_attr, treatment_attr,
                               method=method, depth=depth + 1, max_depth=max_depth,
                               min_split=min_split, min_bucket_t0=min_bucket_t0,
                               min_bucket_t1=min_bucket_t1, mtry=mtry,
                               bins=bins, **kwargs)
        tree.right = build_tree(sub_2, cols, predict_attr, treatment_attr,
                                method=method, depth=depth + 1, max_depth=max_depth,
                                min_split=min_split, min_bucket_t0=min_bucket_t0,
                                min_bucket_t1=min_bucket_t1, mtry=mtry,
                                bins=bins, **kwargs)
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
