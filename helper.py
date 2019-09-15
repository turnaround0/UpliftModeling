import pandas as pd

from dataset import load_json
from plot import plot_all


def print_overview(dataset_name, df, T, Y, ty):
    print('Shape:', df.shape)
    if dataset_name == 'lalonde':
        print('== Sum of each group ==')
        print(Y.groupby(T).sum())
        print('== Count of each group ==')
        print(T.groupby(T).count())
        print('== Average of each group ==')
        avg = Y.groupby(T).sum() / T.groupby(T).count()
        print(avg)
        print('Uplift:', avg[1] - avg[0])
    else:
        count = ty.groupby(ty).count()
        print('== Count of each group ==')
        print(count)
        uplift = count['TR'] / (count['TR'] + count['TN']) - count['CR'] / (count['CR'] + count['CN'])
        print('Uplift:', uplift)


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


def get_uplift(dataset_name, T, Y):
    if dataset_name == 'lalonde':
        avg = Y.groupby(T).sum() / T.groupby(T).count()
        return avg[1] - avg[0]
    else:
        ty = pd.DataFrame({'Y': Y, 'T': T}) \
            .apply(lambda row: ty_assign(row['Y'], row['T']), axis=1)
        count = ty.groupby(ty).count()
        uplift = count['TR'] / (count['TR'] + count['TN']) - count['CR'] / (count['CR'] + count['CN'])
        return uplift


def plot_data(dataset_names):
    for dataset_name in dataset_names:
        print('*** Dataset name:', dataset_name)
        qini_dict = load_json(dataset_name + '_qini')
        var_sel_dict = load_json(dataset_name + '_val_sel')
        plot_all(dataset_name, qini_dict, var_sel_dict)
