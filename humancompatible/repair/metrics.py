import numpy as np
import pandas as pd

from .data_analysis import rdata_analysis
from .coupling_utils import projection_higher


def DisparateImpact(X_test, y_pred):
    dim = X_test.shape[1] - 2
    df_test = pd.DataFrame(
        np.concatenate((X_test, y_pred.reshape(-1, 1)), axis=1),
        columns=[*range(dim)] + ['S', 'W', 'f'],
    )
    num = (
        sum(df_test[(df_test['S'] == 0) & (df_test['f'] == 1)]['W'])
        / sum(df_test[df_test['S'] == 0]['W'])
    )
    den = (
        sum(df_test[(df_test['S'] == 1) & (df_test['f'] == 1)]['W'])
        / sum(df_test[df_test['S'] == 1]['W'])
    )
    return num / den

def DisparateImpact_postprocess(df_test, y_pred_tmp, favorable_label=1):
    df_tmp = df_test.copy()
    df_tmp.insert(loc=0, column='f', value=y_pred_tmp)

    num = (
        df_tmp[(df_tmp['S'] == 0) & (df_tmp['f'] == favorable_label)]['W'].sum()
        / df_tmp[df_tmp['S'] == 0]['W'].sum()
    )
    den = (
        df_tmp[(df_tmp['S'] == 1) & (df_tmp['f'] == favorable_label)]['W'].sum()
        / df_tmp[df_tmp['S'] == 1]['W'].sum()
    )

    return 1.0 if num == den else num / den

def assess_tv(df, coupling_matrix, x_range, x_list, var_list):
    if len(coupling_matrix):
        df_proj = projection_higher(df, coupling_matrix, x_range, x_list, var_list)
    else:
        df_proj = df
    
    rdist = rdata_analysis(df_proj[['X', 'S', 'W']], x_range, 'X')
    return 0.5 * abs(rdist['x_0'] - rdist['x_1']).sum()


