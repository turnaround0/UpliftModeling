import json
from os import path


def num_class(df, predict_attr, treatment_attr):
    """
    Returns the number of Responders and Non-responders in Treatment and Control group
    """
    tr = df[(df[predict_attr] != 0) & (df[treatment_attr] == 1)]  # Responders in Treatment group
    tn = df[(df[predict_attr] == 0) & (df[treatment_attr] == 1)]  # Non-responders in Treatment group
    cr = df[(df[predict_attr] != 0) & (df[treatment_attr] == 0)]  # Responders in Control group
    cn = df[(df[predict_attr] == 0) & (df[treatment_attr] == 0)]  # Non-responders in Control group
    return tr.shape[0], tn.shape[0], cr.shape[0], cn.shape[0]


def split_class(df, predict_attr, treatment_attr):
    tr = df[(df[predict_attr] != 0) & (df[treatment_attr] == 1)]  # Responders in Treatment group
    tn = df[(df[predict_attr] == 0) & (df[treatment_attr] == 1)]  # Non-responders in Treatment group
    cr = df[(df[predict_attr] != 0) & (df[treatment_attr] == 0)]  # Responders in Control group
    cn = df[(df[predict_attr] == 0) & (df[treatment_attr] == 0)]  # Non-responders in Control group
    return tr, tn, cr, cn


def save_json(name, data):
    with open('output/' + name + '.json', 'w') as f:
        json.dump(data, f)


def load_json(name):
    filename = 'output/' + name + '.json'
    if path.exists(filename):
        with open(filename, 'r') as f:
            print('Open success:', filename)
            return json.load(f)
    else:
        print('Not exist', filename)
        return None


def ty_assign(y, t):
    if y == 1 and t == 1:
        return "TR"
    elif y == 0 and t == 1:
        return "TN"
    elif y == 1 and t == 0:
        return "CR"
    elif y == 0 and t == 0:
        return "CN"
    else:
        return None


def t_assign(ty):
    if ty in ("TR", "TN"):
        return 1
    elif ty in ("CR", "CN"):
        return 0
    else:
        return None


def y_assign(ty):
    if ty in ("TR", "CR"):
        return 1
    elif ty in ("TN", "CN"):
        return 0
    else:
        return None


def normalize(input_df, refer_vars=None):
    df = input_df.copy()
    normalize_vars = {}

    for col in df.columns:
        if refer_vars:
            min_max = normalize_vars.get(col)
            if min_max is None:
                continue
            min_val, max_val = min_max
        else:
            count = df[col].drop_duplicates().count()
            if count < 2:
                continue
            min_val = df[col].min()
            max_val = df[col].max()
            normalize_vars[col] = (min_val, max_val)

        df[col] = (df[col] - min_val) / (max_val - min_val)

    return df, normalize_vars


def denormalize(input_df, normalize_vars):
    df = input_df.copy()

    for col in df.columns:
        min_max = normalize_vars.get(col)
        if min_max is None:
            continue

        min_val, max_val = min_max
        df[col] = df[col] * (max_val - min_val) + min_val

    return df
