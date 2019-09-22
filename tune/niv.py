import numpy as np
import pandas as pd


def niv_variable_selection(x, y, t, max_vars):
    """
    NIV variable selection procedure

    WOEi = ln(P(X=i|Y=1) / P(X=i|Y=0))
    IV = sum(((P(X=i|Y=1) - P(X=1|Y=0)) x WOEi)
    NWOE = WOEt - WOEc
    NIV = 100 x sum((P(X=i|Y=1)tP(X=i|/Y=0)c - P(X=i|Y=0)tP(X=i|Y=1)c) x NWOEi)

    Args:
        x: predictor variables of training dataset,
        y: target variables of training dataset,
        t: treatment variables of training dataset,
        max_vars: maximum number of return variables,
    Return:
        (The list of survived variables)
    """
    y1_t = (y == 1) & (t == 1)
    y0_t = (y == 0) & (t == 1)
    y1_c = (y == 1) & (t == 0)
    y0_c = (y == 0) & (t == 0)

    sum_y1_t = sum(y1_t)
    sum_y0_t = sum(y0_t)
    sum_y1_c = sum(y1_c)
    sum_y0_c = sum(y0_c)

    niv_dict = {}
    for col in x.columns:
        df = pd.concat([x[col].rename(col), y1_t.rename('y1_t'), y0_t.rename('y0_t'),
                        y1_c.rename('y1_c'), y0_c.rename('y0_c')], axis=1)
        x_group = df.groupby(x[col])
        x_sum = x_group.sum()

        if sum_y0_t == 0 or sum_y1_t == 0:
            woe_t = 0
        else:
            woe_t = x_sum.apply(lambda r: np.log((r['y1_t'] * sum_y0_t) / (r['y0_t'] * sum_y1_t))
                                if r['y1_t'] > 0 and r['y0_t'] > 0 else 0, axis=1)

        if sum_y0_c == 0 or sum_y1_c == 0:
            woe_c = 0
        else:
            woe_c = x_sum.apply(lambda r: np.log((r['y1_c'] * sum_y0_c) / (r['y0_c'] * sum_y1_c))
                                if r['y1_c'] > 0 and r['y0_c'] > 0 else 0, axis=1)

        nwoe = woe_t - woe_c

        p_x_y1_t = x_sum['y1_t'] / sum_y1_t if sum_y1_t > 0 else 0
        p_x_y0_t = x_sum['y0_t'] / sum_y0_t if sum_y0_t > 0 else 0
        p_x_y1_c = x_sum['y1_c'] / sum_y1_c if sum_y1_c > 0 else 0
        p_x_y0_c = x_sum['y0_c'] / sum_y0_c if sum_y0_c > 0 else 0
        niv_weight = (p_x_y1_t * p_x_y0_c - p_x_y0_t * p_x_y1_c)

        niv_row = 100 * nwoe * niv_weight
        niv = niv_row.sum()
        niv_dict[col] = niv

    s_niv = pd.Series(niv_dict)
    s_selected_niv = s_niv.sort_values(ascending=False)[: max_vars]

    return s_selected_niv.index
