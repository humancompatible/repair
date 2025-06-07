import numpy as np
import pandas as pd

from operator import itemgetter
from itertools import chain
from data_analysis import rdata_analysis


def tmp_generator(gamma_dict, num, q_dict, q_num, L):
    """Compute intermediate gamma and q matrices for iterative coupling updates.

    Args:
        gamma_dict (dict): Dictionary of previously computed gamma matrices.
        num (int): Current iteration index for gamma_dict.
        q_dict (dict): Previously computed q matrices.
        q_num (int): Index in q_dict corresponding to the current q (<=0 for uniform).
        L (int): Step size defining the relation between gamma indices.

    Returns:
        tuple:
            np.matrix: Updated gamma matrix at index num, shape (bin, bin).
            np.matrix: Updated q matrix used for this step, shape (bin, bin).
    """
    bin = gamma_dict[0].shape[0]

    q = np.matrix(np.ones((bin, bin))) if q_num <= 0 else q_dict[q_num]

    tmp_q = np.zeros((bin, bin))
    tmp_gamma = np.zeros((bin, bin))

    for i in range(bin):
        for j in range(bin):
            denom = gamma_dict[num - L].item(i, j)
            tmp_q[i, j] = q.item(i, j) * gamma_dict[num - L - 1].item(i, j) / denom
            tmp_gamma[i, j] = tmp_q[i, j] * gamma_dict[num - 1].item(i, j)

    return np.matrix(tmp_gamma), np.matrix(tmp_q)

def projection(df, coupling_matrix, x_range, x_name, var_list):
    """Project weighted dataset via a 1D coupling for a single feature.

    Given a DataFrame with columns var_list + ['S', 'W', 'Y'], create a repaired DataFrame
    where each original row is split according to coupling_matrix and its weight redistributed.

    Args:
        df (pd.DataFrame): Input data containing columns [*var_list, 'S', 'W', 'Y'].
        coupling_matrix (np.matrix): 1D coupling of shape (bin, bin) stored as a matrix.
        x_range (list): Ordered support values for the feature x_name.
        x_name (str): Name of the feature being repaired.
        var_list (list): List of all feature column names including x_name.

    Returns:
        pd.DataFrame: Repaired DataFrame, aggregated by var_list + ['S', 'Y'], with updated 'W'.
    """
    bin = len(x_range)
    vars_tmp = var_list.copy()
    vars_tmp.remove(x_name)
    vars_tmp = [x_name] + vars_tmp  # place the var that needs to be repaired the first
    
    df = df[vars_tmp + ['S', 'W', 'Y']]
    coup = coupling_matrix.A.reshape(bin, bin)

    df_t = pd.DataFrame(columns=vars_tmp + ['S', 'W', 'Y'])

    for _, row in df.iterrows():
        loc = np.where([x_range[i] == row[x_name] for i in range(bin)])[0][0]
        rows = np.nonzero(coup[loc, :])[0]
        
        sub_dict = {
            x_name: [x_range[r] for r in rows],
            'W': list(coup[loc, rows] / coup[loc, rows].sum() * row['W'])
        }
        sub_dict.update({var: [row[var]] * len(rows) for var in vars_tmp[1:] + ['S', 'Y']})
        
        df_t = pd.concat(
            [df_t, pd.DataFrame(sub_dict, index=rows)],
            ignore_index=True
        )
    
    df_t = df_t.groupby(
        by=list(chain(*[var_list, ['S', 'Y']])),
        as_index=False
    ).sum()

    return df_t[var_list + ['S', 'W', 'Y']]

def projection_higher(df, coupling_matrix, x_range, x_list, var_list):
    """Project weighted dataset via a higher-dimensional coupling for a multi-feature 'X'.

    This function handles cases where 'X' is a tuple of features (x_list). It replaces columns x_list
    with a single 'X' column before applying the coupling.

    Args:
        df (pd.DataFrame): Input data containing columns [*var_list, 'X', 'S', 'W', 'Y'].
                         If x_list are present, they will be dropped.
        coupling_matrix (np.matrix): Coupling of shape (bin, bin) for the aggregated 'X'.
        x_range (list): Ordered support values for 'X' (could be tuples if dim>1).
        x_list (list): List of feature names combined into 'X'.
        var_list (list): List of all feature names including those in x_list.

    Returns:
        pd.DataFrame: Repaired DataFrame with redistributed weights and same columns [*var_list, 'S', 'W', 'Y'].
    """
    if set(x_list).issubset(df.columns):
        df = df.drop(columns=x_list)
    
    bin = len(x_range)
    arg_list = [elem for elem in var_list if elem not in x_list]
    df = df[arg_list + ['X', 'S', 'W', 'Y']]
    coup = coupling_matrix.A.reshape(bin, bin)
    
    df_t = pd.DataFrame(columns=arg_list + ['X', 'S', 'W', 'Y'])
    
    for _, row in df.iterrows():
        loc = np.where([x_range[b] == row['X'] for b in range(bin)])[0][0]

        sub_dict = {
            'X': x_range,
            'W': list(coup[loc, :] / coup[loc, :].sum() * row['W'])
        }
        sub_dict.update({var: [row[var]] * bin for var in arg_list + ['S', 'Y']})

        df_t = pd.concat(
            [df_t, pd.DataFrame(sub_dict, index=range(bin))],
            ignore_index=True
        )
    
    return df_t

def postprocess(
    df, coupling_matrix, x_list, x_range,
    var_list, var_range, clf, thresh
):
    """Apply classifier on repaired feature space to produce final predictions.

    For each unique combination in var_range, build a sub-DataFrame of all possible x_range,
    predict with clf, and aggregate predictions via coupling_matrix to decide the final label.

    Args:
        df (pd.DataFrame): Original data containing var_list + ['S', 'W', 'Y'].
        coupling_matrix (np.matrix): Coupling of shape (bin, bin).
        x_list (list): Names of features combined into 'X' (if len(x_list)>1).
        x_range (list): Support values for the repaired feature(s).
        var_list (list): All feature names including x_list.
        var_range (list): Unique combinations of features in var_list to predict on.
        clf: Fitted classifier with .predict(...) method.
        thresh (float): Threshold for deciding final label.

    Returns:
        np.ndarray: Array of predicted labels aligned with df order.
    """
    dim = len(x_list)
    var_dim = len(var_list)
    bin = len(x_range)
    x_loc = dict(zip(x_range, range(bin)))
    arg_list = [elem for elem in var_list if elem not in x_list]
    coup = coupling_matrix.A1.reshape((bin, bin))
    
    pred_repaired = {}
    
    for v in var_range:
        if var_dim > 1:
            var_tmp = pd.Series({var_list[d]: v[d] for d in range(var_dim)})
            if dim > 1:
                loc = x_loc[tuple(var_tmp[x_list])]
            else:
                loc = x_loc[var_tmp[x_list[0]]]
        else:
            var_tmp = pd.Series({var_list[0]: v})
            loc = x_loc[var_tmp[x_list[0]]]

        sub = pd.DataFrame(x_range, columns=x_list)
        for arg in arg_list:
            sub[arg] = var_tmp[arg]
        sub = sub[var_list]

        total_w = coup[loc, :].sum()
        pred = int(
            np.sum(
                coup[loc, :] / total_w *
                clf.predict(sub.to_numpy().reshape(-1, var_dim))
            ) > thresh
        )
        pred_repaired[v] = pred

    if var_dim > 1:
        keys = list(zip(*[df[c] for c in var_list]))
        return np.array(itemgetter(*keys)(pred_repaired))
    else:
        keys = list(df[var_list[0]])
        return np.array(itemgetter(*keys)(pred_repaired))

def postprocess_bary(
    df, coupling_bary_matrix, x_list, x_range,
    var_list, var_range, clf, thresh
):
    """Apply classifier on barycentric coupling and return TV and predictions.

    Splits DataFrame by S=0/1, applies a barycentric coupling to each subgroup,
    projects distributions, computes total variation, and produces final labels.

    Args:
        df (pd.DataFrame): Original data containing var_list + ['S', 'W', 'Y'].
        coupling_bary_matrix (np.matrix): Barycentric coupling of shape (bin, bin).
        x_list (list): Names of features combined into 'X'.
        x_range (list): Support values for 'X'.
        var_list (list): All feature names including x_list.
        var_range (list): Unique combinations of features in var_list to predict on.
        clf: Fitted classifier with .predict(...) method.
        thresh (float): Threshold for final label decision.

    Returns:
        tuple:
            np.ndarray: Array of predicted labels aligned with original df.
            float: Total variation between S=0 and S=1 projected distributions.
    """
    bin = len(x_range)
    coup_bary = coupling_bary_matrix.A1.reshape((bin,bin))
    s0 = df[df['S'] == 0].copy()
    s1 = df[df['S'] == 1].copy()
    pi0 = len(s0) / len(df)
    pi1 = len(s1) / len(df)
    
    coup0 = np.zeros((bin, bin))
    coup1 = np.zeros((bin, bin))

    for i in range(bin):
        for j in range(bin):
            # assume the distance between every two adjacent x indices is the same
            ind0 = int(pi0 * i + pi1 * j)
            ind1 = int(pi0 * j + pi1 * i)
            coup0[i, ind0] += coup_bary[i, j]
            coup1[i, ind1] += coup_bary[j, i]

    proj0 = projection_higher(
        s0, np.matrix(coup0), x_range, x_list, var_list
    )
    proj1 = projection_higher(
        s1, np.matrix(coup1), x_range, x_list, var_list
    )

    tv = (
        abs(
            rdata_analysis(proj0, x_range, 'X')['x_0']
            - rdata_analysis(proj1, x_range, 'X')['x_1']
        ).sum()
    ) / 2

    s0.insert(loc=0, column='f', value=postprocess(
        s0, np.matrix(coup0), x_list, x_range,
        var_list, var_range, clf, thresh
    ))
    s1.insert(loc=0, column='f', value=postprocess(
        s1, np.matrix(coup1), x_list, x_range,
        var_list, var_range, clf, thresh
    ))

    preds = pd.concat([s0, s1]).sort_index()['f'].to_numpy()
    return preds, tv


