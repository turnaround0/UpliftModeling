import pandas as pd
from imblearn.over_sampling import SMOTE


def over_sampling(X, T, Y):
    new_X = X.copy()
    new_X['T'] = T

    sm = SMOTE(random_state=1234)
    smote_X, smote_Y = sm.fit_resample(new_X, Y)

    smote_X = pd.DataFrame(smote_X, columns=new_X.columns.values)
    smote_Y = pd.DataFrame(smote_Y, columns=['Y'])
    smote_T = smote_X['T'].apply(lambda x: 0 if x < 0.5 else 1)
    smote_X = smote_X.drop(['T'], axis=1)

    return smote_X, smote_T, smote_Y
