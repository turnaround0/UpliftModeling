import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

hillstrom_df = pd.read_csv('Hillstrom.csv')
lalonde_df = pd.read_csv('Lalonde.csv')


def preprocess_data(df, dataset='hillstrom', verbose=True):
    """
    Preprocessing the dataset
     - Use one-hot encoding for categorical features
     - Check the name of the target variable and treatment variable
     - Drop the unused columns
     - Delete the unused data

    Args:
        df: A pandas.DataFrame which have all data of the dataset
        dataset: the name of the dataset
    Return:
        # I recommend to split into the data frames of predictor variables,
        # the target variable, and the treatment variable
        # df: the data frames of predictor variables
        # df['T']: target variables
        # df['Y']: treatment variables
    """
    if dataset in ['hillstrom', 'email']:
        # For Hillstrom dataset, the ‘‘visit’’ target variable was selected
        #   as the target variable of interest and the selected treatment is
        #   the e-mail campaign for women’s merchandise [1]
        # [1] Kane K, Lo VSY, Zheng J. True-lift modeling: Comparison of methods.
        #    J Market Anal. 2014;2:218–238

        # Delete unused data: men's email cases should be removed
        df = df[df.segment != 'Mens E-Mail'].reset_index()
        # Assign Y for target (visit: 0, 1)
        df['Y'] = df['visit']
        # Assign T for treatment (segment: Womens E-Mail, Mens E-Mail (not used), No E-Mail)
        df['T'] = (df['segment'] == 'Womens E-Mail').astype('int64')
        # Drop unused columns from X
        df = df.drop(columns=['conversion', 'spend', 'visit', 'segment', 'index'])
        # One-hot encoding for categorical features
        df = pd.get_dummies(df)

    elif dataset in ['criteo', 'ad']:
        raise NotImplementedError

    elif dataset in ['lalonde', 'job']:
        # Delete unused data: None
        df = df.reset_index()
        # Target variables (RE78: earnings in 1978)
        df['Y'] = df['RE78']
        # Treatment variables (treatment: 0, 1)
        df['T'] = df['treatment']
        # Drop unused columns
        df = df.drop(columns=['treatment', 'RE78', 'index'])
        # One-hot encoding for categorical features
        df = pd.get_dummies(df)

    else:
        raise NotImplementedError

    return df


def performance(pr_y1_t1, pr_y1_t0, y, t, groups=10):
    """
    1. Split the total customers into the given number of groups
    2. Calculate the statistics of each segment

    Args:
        pr_y1_t1: the series (list) of the customer's expected return
        pr_y1_t0: the expected return when a customer is not treated
        y: the observed return of customers
        t: whether each customer is treated or not
        groups: the number of groups (segments). Should be 5, 10, or 20
    Return:
        DataFrame:
            columns:
                'n_y1_t1': the number of treated responders
                'n_y1_t0': the number of not treated responders
                'r_y1_t1': the average return of treated customers
                'r_y1_t0': the average return of not treated customers
                'n_t1': the number of treated customers
                'n_t0': the number of not treated customers
                'uplift': the average uplift (the average treatment effect)
            rows: the index of groups
    """

    ### check valid arguments
    if groups not in [5, 10, 20]:
        raise Exception("uplift: groups must be either 5, 10 or 20")

    ### check for NAs.
    if pr_y1_t1.isnull().values.any():
        raise Exception("uplift: NA not permitted in pr_y1_t1")
    if pr_y1_t0.isnull().values.any():
        raise Exception("uplift: NA not permitted in pr_y1_t0")
    if y.isnull().values.any():
        raise Exception("uplift: NA not permitted in y")
    if t.isnull().values.any():
        raise Exception("uplift: NA not permitted in t")

    ### check valid values for ct
    if set(t) != {0, 1}:
        raise Exception("uplift: t must be either 0 or 1")

    ### check length of arguments
    if not (len(pr_y1_t1) == len(pr_y1_t0) == len(y) == len(t)):
        raise Exception("uplift: arguments pr_y1_t1, pr_y1_t0, y and t must all have the same length")

    # TR(y1_t1): treated responder, CR(y1_t0): controlled responder
    # Uplift = P(TR) - P(CR)
    # Customers ordered by predicted uplift values in descending order are segmented.
    df = pd.DataFrame(data={'t': t, 'y': y,
                            'uplift_rank': (pr_y1_t1 - pr_y1_t0).rank(ascending=False, method='first')})
    df_group = df.groupby(pd.qcut(df['uplift_rank'], groups, labels=range(1, groups + 1)).rename('group'))

    # Get group data
    n_t1 = df_group['t'].sum()
    n_t0 = df_group['t'].count() - n_t1
    n_y1_t1 = df_group.apply(lambda r: r[r['t'] == 1]['y'].sum())
    n_y1_t0 = df_group.apply(lambda r: r[r['t'] == 0]['y'].sum())
    r_y1_t1 = n_y1_t1 / n_t1
    r_y1_t0 = n_y1_t0 / n_t0
    uplift = r_y1_t1 - r_y1_t0

    return pd.DataFrame(data={'n_y1_t1': n_y1_t1, 'n_y1_t0': n_y1_t0,
                              'r_y1_t1': r_y1_t1, 'r_y1_t0': r_y1_t0,
                              'n_t1': n_t1, 'n_t0': n_t0, 'uplift': uplift})


def plot_qini_curve(x, y_list):
    """
    Plot qini curve with multiple Y
    """
    for y in y_list:
        plt.plot(x, y)
    plt.show()


def calc_auuc(decile_width, decile_size, gains):
    """
    Calculate AUUC (Area Under Uplift Curve) and return AUUC value and list
    """
    auuc = 0
    auuc_list = [auuc]
    for i in range(1, decile_size + 1):
        auuc += 0.5 * decile_width * (gains[i] + gains[i - 1])
        auuc_list.append(auuc)
    return auuc, auuc_list


def qini(perf, plotit=True):
    """
    Calculating the incremental gains (y-axis of Qini curve)
     - First, the cumulative sum of the treated and the control groups are
      calculated with respect to the total population in each group at the
      specified decile
     - Afterwards we calculate the percentage of the total amount of people
      (both treatment and control) are present in each decile
    Args:
        perf: A return of the performance function (above)
        plotit: whether draw a plot or not
    Return:
        1. Qini value
        2. return or save the plot if plotit is True
    """
    n_size = len(perf)
    cumsum_r_y1_t1 = perf['n_y1_t1'].cumsum() / perf['n_t1'].cumsum()
    cumsum_r_y1_t0 = perf['n_y1_t0'].cumsum() / perf['n_t0'].cumsum()
    deciles = np.linspace(0, 1, num=n_size + 1)
    decile_width = 1 / n_size

    # Model Incremental gains: first gain is 0
    inc_gains = ([0] + list(cumsum_r_y1_t1 - cumsum_r_y1_t0)) * deciles

    # Overall incremental gains
    overall_inc_gain = perf['n_y1_t1'].sum() / perf['n_t1'].sum() - perf['n_y1_t0'].sum() / perf['n_t0'].sum()

    # Random incremental gains
    random_inc_gains = overall_inc_gain * deciles

    # Compute area under the model incremental gains (uplift) curve
    auuc, auuc_list = calc_auuc(decile_width, n_size, inc_gains)

    # Compute area under the random incremental gains curve
    auuc_rand, auuc_rand_list = calc_auuc(decile_width, n_size, random_inc_gains)

    # Compute the difference between the areas (Qini coefficient)
    qini_coefficient = auuc - auuc_rand

    # Qini 30%, Qini 10%
    n_30p = int(n_size * 0.3)
    n_10p = int(n_size * 0.1)
    qini_30p = auuc_list[n_30p] - auuc_rand_list[n_30p]
    qini_10p = auuc_list[n_10p] - auuc_rand_list[n_10p]

    # Plot incremental gains curve
    if plotit:
        plot_qini_curve(deciles, [inc_gains, random_inc_gains])

    return {
        'qini': qini_coefficient,
        'inc_gains': inc_gains,
        'random_inc_gains': random_inc_gains,
        'auuc_list': auuc_list,
        'auuc_rand_list': auuc_rand_list,
        'qini_30p': qini_30p,
        'qini_10p': qini_10p,
    }


def parameter_tuning(fit_mdl, pred_mdl, data, search_space):
    """
    Given a model, search all combination of parameter sets and find
    the best parameter set

    Args:
        fit_mdl: model function
        pred_mdl: predict function of fit_mdl
        data:
            {
                "x_train": predictor variables of training dataset,
                "y_train": target variables of training dataset,
                "t_train": treatment variables of training dataset,
                "x_test": predictor variables of test (usually, validation) dataset,
                "y_test": target variables of test (usually, validation) dataset,
                "t_test": treatment variables of test (usually, validation) dataset,
            }
        search_space:
            {
                parameter_name: [search values]
            }
    Return:
        The best parameter set
    """
    max_q = -float('inf')
    best_mdl = None

    # Grid search: find all possible cases
    keys = search_space.keys()
    n_space = [len(search_space[key]) for key in keys]
    n_iter = np.prod(n_space)

    best_params = None
    for i in range(n_iter):
        # Make grid search params
        params = {}
        for idx, key in enumerate(keys):
            params[key] = search_space[key][i % n_space[idx]]
            i = int(i / n_space[idx])

        # Build model and predict
        mdl = fit_mdl(data['x_train'], data['y_train'], data['t_train'], **params)
        pred = pred_mdl(mdl, newdata=data['x_test'], y=data['y_test'], ct=data['t_test'])

        # Calculate qini value
        try:
            perf = performance(pred['pr_y1_t1'], pred['pr_y1_t0'], data['y_test'], data['t_test'])
        except Exception as e:
            print(e)
            continue

        q = qini(perf, plotit=False)['qini']
        if q > max_q:
            max_q = q
            best_mdl = mdl
            best_params = params

    return best_mdl, best_params


def wrapper(fit_mdl, pred_mdl, data, params=None,
            best_models=None, drop_variables=None, qini_values=None):
    """
    General wrapper approach

    Args:
        fit_mdl: model function
        pred_mdl: predict function of fit_mdl
        data:
            {
                "x_train": predictor variables of training dataset,
                "y_train": target variables of training dataset,
                "t_train": treatment variables of training dataset,
                "x_test": predictor variables of test (usually, validation) dataset,
                "y_test": target variables of test (usually, validation) dataset,
                "t_test": treatment variables of test (usually, validation) dataset,
            }
        params: given parameters for model
        best_models:
        drop_variables: don't check performance for those variables
        qini_values:
    Return:
        (A list of best models, The list of dropped variables)
    """
    if best_models is None:
        best_models = []
    if drop_variables is None:
        drop_variables = []
    if qini_values is None:
        qini_values = []
    if params is None:
        params = {}

    max_q = -float('inf')
    drop_var = None
    best_mdl = None

    # Check performance drop for each predictor variable
    variables = data['x_train'].columns
    for var in variables:
        if var in drop_variables:
            continue

        # Build and train model
        x = data['x_train'].drop(drop_variables + [var], axis=1)
        mdl = fit_mdl(x, data['y_train'], data['t_train'], **params)

        # Predict by the model
        x = data['x_test'].drop(drop_variables + [var], axis=1)
        pred = pred_mdl(mdl, newdata=x, y=data['y_test'], t=data['t_test'])

        # Calculate qini value
        perf = performance(pred['pr_y1_t1'], pred['pr_y1_t0'], data['y_test'], data['t_test'])
        q = qini(perf, plotit=False)['qini']
        if q > max_q:
            max_q = q
            drop_var = var
            best_mdl = mdl

    best_models.append(best_mdl)
    drop_variables.append(drop_var)
    qini_values.append(max_q)

    if len(variables) == len(drop_variables) + 1:
        left_vars = [var for var in variables if var not in drop_variables]
        return best_models, drop_variables + left_vars, qini_values
    else:
        return wrapper(fit_mdl, pred_mdl, data, params=params,
                       best_models=best_models, drop_variables=drop_variables,
                       qini_values=qini_values)


def tma(x, y, t, method=LogisticRegression, **kwargs):
    """Training a model according to the "Two Model Approach"
    (a.k.a. "Separate Model Approach")
    The default model is General Linear Model (GLM)

    Source: "Incremental Value Modeling" (Hansotia, 2002)

    Args:
        x: A data frame of predictors.
        y: A binary response (numeric) vector.
        t: A binary response (numeric) representing the treatment assignment
            (coded as 0/1).
        method: A sklearn model specifying which classification or regression
            model to use. This should be a method that can handle a
            multinominal class variable.

    Return:
        Dictionary: A dictionary of two models. One for the treatment group,
            one for the control group.

            {
                'model_treat': a model for the treatment group,
                'model_control': a model for the control group
            }

    """
    return {
        'model_treat': method(**kwargs).fit(x[t == 1], y[t == 1]),
        'model_control': method(**kwargs).fit(x[t == 0], y[t == 0])
    }


def predict_tma(obj, newdata, **kwargs):
    """Predictions according to the "Two Model Approach"
    (a.k.a. "Separate Model Approach")

    For each instance in newdata two predictions are made:
    1) What is the probability of a person responding when treated?
    2) What is the probability of a person responding when not treated
      (i.e. part of control group)?

    Source: "Incremental Value Modeling" (Hansotia, 2002)

    Args:
        obj: A dictionary of two models.
            One for the treatment group, one for the control group.
        newdata: A data frame containing the values at which predictions
            are required.

    Return:
        DataFrame: A data frame with predicted returns for when the customers
            are treated and for when they are not treated.
            'pr_y1_t1': when treated, 'pr_y1_t0': when not treated
    """
    # LogisticRegression: use predict_proba and return result[:, 1] for True class
    # LinearRegression: use predict
    if isinstance(obj['model_treat'], LogisticRegression):
        return pd.DataFrame(data={
            'pr_y1_t1': obj['model_treat'].predict_proba(newdata, **kwargs)[:, 1],
            'pr_y1_t0': obj['model_control'].predict_proba(newdata, **kwargs)[:, 1]
        })
    else:
        return pd.DataFrame(data={
            'pr_y1_t1': obj['model_treat'].predict(newdata, **kwargs),
            'pr_y1_t0': obj['model_control'].predict(newdata, **kwargs)
        })


class Node(object):
    """
    Node of decision tree
    """
    def __init__(self, attribute, threshold):
        self.attr = attribute
        self.thres = threshold
        self.left = None
        self.right = None
        self.leaf = False
        self.predict = None


def select_threshold(df, attribute, predict_attr):
    """
    Select the threshold of the attribute to split
    The threshold chosen splits the test data such that information gain is maximized
    """
    values = df[attribute].astype('float').drop_duplicates().sort_values().reset_index(drop=True)

    max_ig = float("-inf")
    thres_val = 0
    # try all threshold values that are half-way between successive values in this sorted list
    for i in range(0, len(values) - 1):
        thres = (values[i] + values[i + 1]) / 2
        ig = info_gain(df, attribute, predict_attr, thres)
        if ig > max_ig:
            max_ig = ig
            thres_val = thres

    # Return the threshold value that maximizes information gained
    return thres_val


def info_entropy(df, predict_attr):
    """
    Calculate info content (entropy) of the test data
    """
    # Data frame and number of positive/negatives examples in the data
    p = float(df[df[predict_attr] == 1].shape[0])
    n = float(df[df[predict_attr] == 0].shape[0])

    # Calculate entropy
    if p == 0 or n == 0:
        return 0
    else:
        return -(p / (p + n)) * math.log(p / (p + n), 2) - (n / (p + n)) * math.log(n / (p + n), 2)


def remainder(df, df_subsets, predict_attr):
    """
    Calculates the weighted average of the entropy after an attribute test
    """
    # number of test data
    num_data = df.shape[0]

    # Calculate the weighted average of the entropy of subsets
    r = float(0)
    for df_sub in df_subsets:
        if df_sub.shape[0] > 1:
            r += float(df_sub.shape[0] / num_data) * info_entropy(df_sub, predict_attr)
    return r


def info_gain(df, attribute, predict_attr, threshold):
    """
    Calculates the information gain from the attribute test based on a given threshold
    Note: thresholds can change for the same attribute over time
    """
    sub_1 = df[df[attribute] <= threshold]
    sub_2 = df[df[attribute] > threshold]

    # Determine information content, and subtract remainder of attributes from it
    ig = info_entropy(df, predict_attr) - remainder(df, [sub_1, sub_2], predict_attr)
    return ig


def num_class(df, predict_attr):
    """
    Returns the number of positive and negative data
    """
    return df[df[predict_attr] == 1].shape[0], df[df[predict_attr] == 0].shape[0]


def choose_attr(df, attributes, predict_attr):
    """
    Chooses the attribute and its threshold with the highest info gain
    from the set of attributes
    """
    max_info_gain = float("-inf")
    best_attr = None
    threshold = 0

    # Test each attribute (note attributes maybe be chosen more than once)
    for attr in attributes:
        thres = select_threshold(df, attr, predict_attr)
        ig = info_gain(df, attr, predict_attr, thres)
        if ig > max_info_gain:
            max_info_gain = ig
            best_attr = attr
            threshold = thres

    return best_attr, threshold


def build_tree(df, cols, predict_attr):
    """
    Builds the Decision Tree based on training data, attributes to train on,
    and a prediction attribute
    """
    # Get the number of positive and negative examples in the training data
    p, n = num_class(df, predict_attr)

    # If train data has all positive or all negative values or less than the
    # given minimum number of split. Then we have reached the end of our tree
    if p == 0 or n == 0 or (p + n) < 100:
        # Create a leaf node indicating it's prediction
        leaf = Node(None, None)
        leaf.leaf = True
        leaf.predict = p / (p + n)
        return leaf
    else:
        # Determine attribute and its threshold value with the highest
        # information gain
        best_attr, threshold = choose_attr(df, cols, predict_attr)

        # Create internal tree node based on attribute and it's threshold
        tree = Node(best_attr, threshold)
        sub_1 = df[df[best_attr] <= threshold]
        sub_2 = df[df[best_attr] > threshold]

        # Recursively build left and right subtree
        tree.left = build_tree(sub_1, cols, predict_attr)
        tree.right = build_tree(sub_2, cols, predict_attr)
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


def test_predictions(root, df, target_attr='y'):
    """
    Given a set of data, make a prediction for each instance using the Decision Tree
    """
    prediction = []
    for index, row in df.iterrows():
        prediction.append(predict(root, row))

    pred_df = pd.Series(prediction)
    return pred_df


def main():
    ### Hyper parameters ###
    dataset = 'hillstrom'
    seed = 1234
    n_fold = 5
    models = {
        'tma': {'model': tma, 'predict': predict_tma}
    }
    search_space = {
        'method': [LogisticRegression],
        'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
        'penalty': ['none', 'l2'],
        'tol': [1e-2, 1e-3, 1e-4],
        'C': [1e6, 1e3, 1, 1e-3, 1e-6]
    }

    # Preprocessing data & K fold validation
    if dataset == 'hillstrom':
        df_all = preprocess_data(hillstrom_df, dataset)
        fold_split = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed).split(df_all, df_all['Y'])
    elif dataset == 'lalonde':
        df_all = preprocess_data(lalonde_df, dataset)
        fold_split = KFold(n_splits=n_fold, shuffle=True, random_state=seed).split(df_all)
    else:
        assert ()

    for model_name in models:
        qini_list = []

        for train_index, test_index in fold_split:
            df_train = df_all.reindex(train_index).reset_index(drop=True)
            df_test = df_all.reindex(test_index).reset_index(drop=True)

            # Data split: 2/3 tuning dataset, 1/3 validation dataset
            stratify = pd.DataFrame([df_train['Y'], df_train['T']]).T
            df_tune, df_val = train_test_split(df_train, test_size=0.33, random_state=seed, stratify=stratify)

            # Variable selection (General wrapper approach)
            data_dict = {
                "x_train": df_tune.drop(columns=['Y', 'T']).reset_index(drop=True),
                "y_train": df_tune['Y'].reset_index(drop=True),
                "t_train": df_tune['T'].reset_index(drop=True),
                "x_test": df_val.drop(columns=['Y', 'T']).reset_index(drop=True),
                "y_test": df_val['Y'].reset_index(drop=True),
                "t_test": df_val['T'].reset_index(drop=True),
            }

            model_method = search_space.get('method', None)
            params = {
                'method': None if model_method is None else model_method[0],
            }
            if params['method'] == LogisticRegression:
                solver = search_space.get('solver', None)
                params['solver'] = None if solver is None else solver[0]

            _, drop_vars, qini_values = wrapper(models[model_name]['model'],  models[model_name]['predict'],
                                                data_dict, params=params)

            best_qini = max(qini_values)
            best_idx = qini_values.index(best_qini)
            best_drop_vars = drop_vars[: best_idx]

            data_dict['x_train'].drop(best_drop_vars, axis=1, inplace=True)
            data_dict['x_test'].drop(best_drop_vars, axis=1, inplace=True)
            df_train.drop(best_drop_vars, axis=1, inplace=True)
            df_test.drop(best_drop_vars, axis=1, inplace=True)

            # Parameter tuning
            _, best_params = parameter_tuning(models[model_name]['model'], models[model_name]['predict'],
                                              data_dict, search_space=search_space)

            # Train model and predict
            x_train = df_all.drop(columns=['T', 'Y']).reindex(train_index).reset_index(drop=True)
            y_train = df_all['Y'].reindex(train_index).reset_index(drop=True)
            t_train = df_all['T'].reindex(train_index).reset_index(drop=True)

            x_test = df_all.drop(columns=['T', 'Y']).reindex(test_index).reset_index(drop=True)
            t_test = df_all['Y'].reindex(test_index).reset_index(drop=True)
            y_test = df_all['T'].reindex(test_index).reset_index(drop=True)

            model = models[model_name]['model'](x_train, y_train, t_train, **best_params)
            pred = models[model_name]['predict'](model, x_test)

            # Calculate qini value
            perf = performance(pred['pr_y1_t1'], pred['pr_y1_t0'], y_test, t_test)
            q = qini(perf)
            qini_list.append(q['qini'])

        print("Model: {}\n".format(model_name))
        print("Tuning space: \n")
        for key, val in search_space.items():
            print("    '{}': {}\n".format(key, val))
        print("Seed: {}\n".format(seed))
        print('Qini values: ', qini_list)
        print("Qini value: mean = {}, std = {}\n\n".format(np.mean(qini_list), np.std(qini_list)))


def main_tree():
    ### Hyper parameters ###
    dataset = 'hillstrom'
    seed = 1234
    n_fold = 5
    models = {
        'tma': {'model': tma, 'predict': predict_tma}
    }
    search_space = {
        'method': [LogisticRegression],
        'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
        'penalty': ['none', 'l2'],
        'tol': [1e-2, 1e-3, 1e-4],
        'C': [1e6, 1e3, 1, 1e-3, 1e-6]
    }

    # Preprocessing data & K fold validation
    if dataset == 'hillstrom':
        df_all = preprocess_data(hillstrom_df, dataset)
        fold_split = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed).split(df_all, df_all['Y'])
    elif dataset == 'lalonde':
        df_all = preprocess_data(lalonde_df, dataset)
        fold_split = KFold(n_splits=n_fold, shuffle=True, random_state=seed).split(df_all)
    else:
        assert ()

    for model_name in models:
        qini_list = []

        for train_index, test_index in fold_split:
            # Drop history column due to too slow learning speed
            df_train = df_all.reindex(train_index).reset_index(drop=True).drop(['history'], axis=1)
            df_test = df_all.reindex(test_index).reset_index(drop=True).drop(['history'], axis=1)

            t_test = df_all['Y'].reindex(test_index).reset_index(drop=True)
            y_test = df_all['T'].reindex(test_index).reset_index(drop=True)

            assert ((df_train.columns == df_test.columns).all())

            # Build decision tree and predict on Two Model approach
            attributes = [col for col in df_train.columns if col != 'Y']
            root_t = build_tree(df_train[df_train['T'] == 1].reset_index(drop=True), attributes, 'Y')
            pred_t = test_predictions(root_t, df_test, target_attr='Y')

            root_c = build_tree(df_train[df_train['T'] == 0].reset_index(drop=True), attributes, 'Y')
            pred_c = test_predictions(root_c, df_test, target_attr='Y')
            print('pred: {}'.format(pred_t[: 5]))
            print('pred: {}'.format(pred_c[: 5]))

            perf = performance(pred_t, pred_c, y_test, t_test)
            print(perf[:5])
            q = qini(perf)
            print(q)
            qini_list.append(q['qini'])

        print("Model: {}\n".format(model_name))
        print("Tuning space: \n")
        for key, val in search_space.items():
            print("    '{}': {}\n".format(key, val))
        print("Seed: {}\n".format(seed))
        print('Qini values: ', qini_list)
        print("Qini value: mean = {}, std = {}\n\n".format(np.mean(qini_list), np.std(qini_list)))

main_tree()
