#import math
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from operator import itemgetter

from itertools import chain


def normialise(tem_dist):
    total = sum(tem_dist)
    return [val / total for val in tem_dist]

def tmp_generator(gamma_dict, num, q_dict, q_num, L):
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

def newton(fun, dfun, a, stepmax, tol):
    if abs(fun(a)) <= tol:
        return a
    
    for _ in range(1, stepmax + 1):
        b = a - fun(a) / dfun(a)
        if abs(fun(b)) <= tol:
            return b
        a = b
    
    return b 

# simplist
def baseline(C, e, px, ptx, K):
    bin = len(px)
    bbm1 = np.matrix(np.ones(bin)).T
    xi = np.exp(-C / e)

    gamma_classic = {}
    gamma_classic[0] = np.matrix(xi + 1.0e-9)

    for repeat in range(K):
        gamma_classic[1 + 2 * repeat] = (
            np.matrix(np.diag((px / (gamma_classic[2 * repeat] @ bbm1)).A1))
            @ gamma_classic[2 * repeat]
        )
        gamma_classic[2 + 2 * repeat] = (
            gamma_classic[1 + 2 * repeat]
            @ np.matrix(np.diag((ptx / (gamma_classic[1 + 2 * repeat].T @ bbm1)).A1))
        )

    return gamma_classic[2 * K]

# our method | total repair
def total_repair(C, e, px, ptx, V, K):
    bin = len(px)
    bbm1 = np.matrix(np.ones(bin)).T
    I = np.where(~(V == 0))[0].tolist()
    xi = np.exp(-C / e)

    gamma_dict = {}
    gamma_dict[0] = np.matrix(xi + 1.0e-9)
    gamma_dict[1] = np.matrix(np.diag((px / (gamma_dict[0] @ bbm1)).A1)) @ gamma_dict[0]
    gamma_dict[2] = gamma_dict[1] @ np.matrix(
        np.diag((ptx / (gamma_dict[1].T @ bbm1)).A1)
    )

    # step 3
    J = np.where(~((gamma_dict[2].T @ V).A1 == 0))[0].tolist()
    gamma_dict[3] = np.copy(gamma_dict[2])

    for j in J:
        fun = lambda z: sum(
            gamma_dict[2].item(i, j) * V.item(i) * np.exp(z * V.item(i)) for i in I
        )
        dfun = lambda z: sum(
            gamma_dict[2].item(i, j) * (V.item(i) ** 2) * np.exp(z * V.item(i)) for i in I
        )
        nu = newton(fun, dfun, 0, stepmax=50, tol=1.0e-9) 
        for i in I:
            gamma_dict[3][i, j] = np.exp(nu * V.item(i)) * gamma_dict[2].item(i, j)

    gamma_dict[3] = np.matrix(gamma_dict[3])

    #=========================
    L = 3
    q_dict = {}

    for loop in range(1,K):
        if np.any(gamma_dict[(loop - 1) * L + 1] == 0):
            break

        tmp, q_dict[(loop - 1) * L + 1] = tmp_generator(
            gamma_dict, loop * L + 1, q_dict, (loop - 2) * L + 1, L
        )
        gamma_dict[loop * L + 1] = (
            np.matrix(np.diag((px / (tmp @ bbm1)).A1)) @ tmp
        )

        if np.any(gamma_dict[(loop - 1) * L + 2] == 0):
            break

        tmp, q_dict[(loop - 1) * L + 2] = tmp_generator(
            gamma_dict, loop * L + 2, q_dict, (loop - 2) * L + 2, L
        )
        gamma_dict[loop * L + 2] = tmp @ np.matrix(
            np.diag((ptx / (tmp.T @ bbm1)).A1)
        )

        # step 3
        if np.any(gamma_dict[(loop - 1) * L + 3] == 0):
            break

        tmp, q_dict[(loop - 1) * L + 3] = tmp_generator(
            gamma_dict, loop * L + 3, q_dict, (loop - 2) * L + 3, L
        )
        J = np.where(~((abs(np.matrix(tmp).T @ V).A1) <= 1.0e-9))[0].tolist()
        gamma_dict[loop * L + 3] = np.copy(tmp)

        for j in J:
            fun = lambda z: sum(
                tmp.item(i, j) * V.item(i) * np.exp(z * V.item(i)) for i in I
            )
            dfun = lambda z: sum(
                tmp.item(i, j) * (V.item(i) ** 2) * np.exp(z * V.item(i)) for i in I
            )
            nu = newton(fun, dfun, 0, stepmax=50, tol=1.0e-9)
            for i in I:
                gamma_dict[loop * L + 3][i, j] = (
                    np.exp(nu * V.item(i)) * tmp.item(i, j)
                )

        gamma_dict[loop * L + 3] = np.matrix(gamma_dict[loop * L + 3])

    return gamma_dict[loop * L + 3]

# our method | partial repair
def partial_repair(C, e, px, ptx, V, theta_scale, K):
    bin = len(px)
    bbm1 = np.matrix(np.ones(bin)).T
    I = np.where(~(V == 0))[0].tolist()
    xi = np.exp(-C / e)
    theta = bbm1 * theta_scale

    gamma_dict = {}
    gamma_dict[0] = np.matrix(xi + 1.0e-9)
    gamma_dict[1] = np.matrix(np.diag((px / (gamma_dict[0] @ bbm1)).A1)) @ gamma_dict[0]
    gamma_dict[2] = gamma_dict[1] @ np.matrix(
        np.diag((ptx / (gamma_dict[1].T @ bbm1)).A1)
    )

    # step 3
    Jplus = np.where(~((gamma_dict[2].T @ V).A1 <= theta.A1))[0].tolist()
    Jminus = np.where(~((gamma_dict[2].T @ V).A1 >= -theta.A1))[0].tolist()
    gamma_dict[3] = np.copy(gamma_dict[2])

    for j in Jplus:
        fun = lambda z: sum(
            gamma_dict[2].item(i, j) * V.item(i) * np.exp(-z * V.item(i)) for i in I
        ) - theta.item(j)
        dfun = lambda z: -sum(
            gamma_dict[2].item(i, j) * (V.item(i) ** 2) * np.exp(-z * V.item(i)) for i in I
        )
        nu = newton(fun, dfun, 0, stepmax=50, tol=1.0e-9)
        for i in I:
            gamma_dict[3][i, j] = np.exp(-nu * V.item(i)) * gamma_dict[2].item(i, j)
    
    for j in Jminus:
        fun = lambda z: sum(
            gamma_dict[2].item(i, j) * V.item(i) * np.exp(-z * V.item(i)) for i in I
        ) + theta.item(j)
        dfun = lambda z: -sum(
            gamma_dict[2].item(i, j) * (V.item(i) ** 2) * np.exp(-z * V.item(i)) for i in I
        )
        nu = newton(fun, dfun, 0, stepmax=50, tol=1.0e-9)
        for i in I:
            gamma_dict[3][i, j] = np.exp(-nu * V.item(i)) * gamma_dict[2].item(i, j)

    gamma_dict[3] = np.matrix(gamma_dict[3])

    #=========================
    L = 3
    q_dict = {}

    for loop in range(1, K):
        if np.any(gamma_dict[(loop - 1) * L + 1] == 0):
            break

        tmp, q_dict[(loop - 1) * L + 1] = tmp_generator(
            gamma_dict, loop * L + 1, q_dict, (loop - 2) * L + 1, L
        )
        gamma_dict[loop * L + 1] = (
            np.matrix(np.diag((px / (tmp @ bbm1)).A1)) @ tmp
        )

        if np.any(gamma_dict[(loop - 1) * L + 2] == 0):
            break

        tmp, q_dict[(loop - 1) * L + 2] = tmp_generator(
            gamma_dict, loop * L + 2, q_dict, (loop - 2) * L + 2, L
        )
        gamma_dict[loop * L + 2] = tmp @ np.matrix(
            np.diag((ptx / (tmp.T @ bbm1)).A1)
        )

        # step 3
        if np.any(gamma_dict[(loop - 1) * L + 3] == 0):
            break
        
        tmp, q_dict[(loop - 1) * L + 3] = tmp_generator(
            gamma_dict, loop * L + 3, q_dict, (loop - 2) * L + 3, L
        )
        Jplus = np.where(~((np.matrix(tmp).T @ V).A1 <= theta.A1))[0].tolist()
        Jminus = np.where(~((np.matrix(tmp).T @ V).A1 >= -theta.A1))[0].tolist()
        gamma_dict[loop * L + 3] = np.copy(tmp)

        for j in Jplus:
            fun = lambda z: sum(
                tmp.item(i, j) * V.item(i) * np.exp(-z * V.item(i)) for i in I
            ) - theta.item(j)
            dfun = lambda z: -sum(
                tmp.item(i, j) * (V.item(i) ** 2) * np.exp(-z * V.item(i)) for i in I
            )
            nu = newton(fun, dfun, 0, stepmax=50, tol=1.0e-9)
            for i in I:
                gamma_dict[loop * L + 3][i, j] = (
                    np.exp(-nu * V.item(i)) * tmp.item(i, j)
                )
        
        for j in Jminus:
            fun = lambda z: sum(
                tmp.item(i, j) * V.item(i) * np.exp(-z * V.item(i)) for i in I
            ) + theta.item(j)
            dfun = lambda z: -sum(
                tmp.item(i, j) * (V.item(i) ** 2) * np.exp(-z * V.item(i)) for i in I
            )
            nu = newton(fun, dfun, 0, stepmax=50, tol=1.0e-9)
            for i in I:
                gamma_dict[loop * L + 3][i, j] = (
                    np.exp(-nu * V.item(i)) * tmp.item(i, j)
                )

        gamma_dict[loop * L + 3] = np.matrix(gamma_dict[loop * L + 3])

    return gamma_dict[loop * L + 3]

def empirical_distribution(sub, x_range):
    bin = len(x_range)
    distribution = np.zeros(bin)
    for i in range(bin):
        subset = sub[sub['X'] == x_range[i]]
        if subset.shape[0] > 0:
            distribution[i] = sum(subset['W'])

    return distribution / sum(distribution) if sum(distribution) > 0 else distribution

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
    
def rdata_analysis(rdata, x_range, x_name):
    rdist = {}

    pivot = pd.pivot_table(
        rdata, index=x_name, values=['W'],
        aggfunc=[np.sum], observed=False
    )[('sum', 'W')]
    
    total = sum(pivot[i] for i in x_range)
    rdist['x'] = np.array([pivot[i] for i in x_range]) / total

    if (rdata['S'] == 0).any():
        pivot0 = pd.pivot_table(
            rdata[rdata['S'] == 0], index=x_name, values=['W'],
            aggfunc=[np.sum], observed=False
        )[('sum', 'W')]

        total0 = sum(pivot0[i] if i in pivot0.index else 0 for i in x_range)
        rdist['x_0'] = np.array(
            [pivot0[i] if i in pivot0.index else 0 for i in x_range]
        ) / total0

    if (rdata['S'] == 1).any():
        pivot1 = pd.pivot_table(
            rdata[rdata['S'] == 1], index=x_name, values=['W'],
            aggfunc=[np.sum], observed=False
        )[('sum', 'W')]

        total1 = sum(pivot1[i] if i in pivot1.index else 0 for i in x_range)
        rdist['x_1'] = np.array(
            [pivot1[i] if i in pivot1.index else 0 for i in x_range]
        ) / total1
        
    return rdist

def c_generate(x_range):
    bin = len(x_range)
    C = np.zeros((bin, bin))

    for i in range(bin):
        for j in range(bin):
            C[i, j] = abs(x_range[i] - x_range[j])

    return C

def c_generate_higher(x_range, weight):
    bin = len(x_range)
    dim = len(x_range[0])
    C = np.zeros((bin, bin))

    for i in range(bin):
        for j in range(bin):
            C[i, j] = sum(
                weight[d] * abs(x_range[i][d] - x_range[j][d])
                for d in range(dim)
            )
    
    return C

def projection(df, coupling_matrix, x_range, x_name, var_list):
    bin = len(x_range)
    vars_tmp = var_list.copy()
    vars_tmp.remove(x_name)
    vars_tmp = [x_name] + vars_tmp # place the var that needs to be repaired the first
    
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

def postprocess_bary(
    df, coupling_bary_matrix, x_list, x_range,
    var_list, var_range, clf, thresh
):
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

def assess_tv(df, coupling_matrix, x_range, x_list, var_list):
    if len(coupling_matrix):
        df_proj = projection_higher(df, coupling_matrix, x_range, x_list, var_list)
    else:
        df_proj = df
    
    rdist = rdata_analysis(df_proj[['X', 'S', 'W']], x_range, 'X')
    return 0.5 * abs(rdist['x_0'] - rdist['x_1']).sum()
