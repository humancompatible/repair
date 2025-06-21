import numpy as np
import pandas as pd

from data_analysis import rdata_analysis
from coupling_utils import projection_higher


def DisparateImpact(X_test, y_pred):
    """Compute Disparate Impact: P(f=1|S=0)/P(f=1|S=1), weighted by 'W'.

    Args:
        X_test (ndarray): Shape (n, d+2), columns [..., 'S', 'W'].
        y_pred (ndarray): Predicted labels, shape (n,).

    Returns:
        float: DI ratio.
    """
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
    """Compute Disparate Impact on a DataFrame after inserting predictions as column 'f'.

    Args:
        df_test (DataFrame): Contains 'S', 'W', ... columns.
        y_pred_tmp (ndarray): Predicted labels, shape (n,).
        favorable_label (int): Label to treat as "positive". Defaults to 1.

    Returns:
        float: DI ratio, or 1.0 if groups match exactly.
    """
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
    """Return total variation between S=0 and S=1 distributions (weighted).

    If a coupling is given, first project via projection_higher; otherwise use original df.
    Then compute 0.5 * SUM|p0 - p1| over support x_range.

    Args:
        df (DataFrame): Must contain ['X','S','W'] + var_list.
        coupling_matrix (matrix or array): Shape (len(x_range), len(x_range)), or empty.
        x_range (list): Ordered support values for the repaired variable.
        x_list (list): Column name(s) to repair (univariate if len=1).
        var_list (list): All column names defining unique rows after projection.

    Returns:
        float: Total variation distance.
    """
    if len(coupling_matrix):
        df_proj = projection_higher(df, coupling_matrix, x_range, x_list, var_list)
    else:
        df_proj = df
    
    rdist = rdata_analysis(df_proj[['X', 'S', 'W']], x_range, 'X')
    return 0.5 * abs(rdist['x_0'] - rdist['x_1']).sum()

def newton(fun, dfun, a, stepmax, tol):
    if abs(fun(a)) <= tol:
        return a
    
    for _ in range(1, stepmax + 1):
        b = a - fun(a) / dfun(a)
        if abs(fun(b)) <= tol:
            return b
        a = b
    
    return b
