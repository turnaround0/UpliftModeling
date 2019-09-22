import pandas as pd
from utils.utils import ty_assign


def preprocess_data(df, dataset='hillstrom', verbose=True):
    # For Hillstrom dataset, the ‘‘visit’’ target variable was selected
    #   as the target variable of interest and the selected treatment is
    #   the e-mail campaign for women’s merchandise [1]
    # [1] Kane K, Lo VSY, Zheng J. True-lift modeling: Comparison of methods.
    #    J Market Anal. 2014;2:218–238
    dataset = dataset.lower()
    if dataset in ('hillstrom', 'email'):
        columns = df.columns
        for col in columns:
            if df[col].dtype != object:
                continue
            df = pd.concat(
                [df, pd.get_dummies(df[col],
                                    prefix=col,
                                    drop_first=False)],
                axis=1)
            df.drop([col], axis=1, inplace=True)

        df.columns = [col.replace('-', '').replace(' ', '_').lower()
                      for col in df.columns]
        df = df[df.segment_mens_email == 0]
        df.index = range(len(df))
        df.drop(['segment_mens_email',
                 'segment_no_email',
                 'conversion',
                 'spend'], axis=1, inplace=True)

        y_name = 'visit'
        t_name = 'segment_womens_email'
    elif dataset in ['criteo', 'ad']:
        df = df.fillna(0)
        y_name = 'y'
        t_name = 'treatment'
        # T(0.0, 1.0) type is float, but it should be int for tree.
        df[t_name] = df[t_name].astype('int')
    elif dataset == 'lalonde':
        y_name = 'RE78'
        t_name = 'treatment'
    else:
        raise NotImplementedError

    df['Y'] = df[y_name]
    df.drop([y_name], axis=1, inplace=True)
    df['T'] = df[t_name]
    df.drop([t_name], axis=1, inplace=True)

    return df


def assign_data(df):
    s_y = df['Y']
    s_t = df['T']
    s_x = df.drop(['Y', 'T'], axis=1)

    s_ty = pd.DataFrame({'Y': s_y, 'T': s_t}) \
        .apply(lambda row: ty_assign(row['Y'], row['T']), axis=1)

    return s_x, s_y, s_t, s_ty
