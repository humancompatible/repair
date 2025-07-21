import numpy as np
import pandas as pd


def rdata_analysis(rdata, x_range, x_name):
    """Return overall and S=0/1 weighted distributions over x_range.

    Args:
        rdata (DataFrame): Must contain columns [x_name, 'S', 'W'].
        x_range (list): Ordered support values for the variable x_name.
        x_name (str): Column name in rdata whose distribution is computed.

    Returns:
        dict: {
            'x' (np.ndarray): overall distribution,
            'x_0' (np.ndarray): distribution for S=0 (if present),
            'x_1' (np.ndarray): distribution for S=1 (if present)
        }
    """
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

def empirical_distribution(sub, x_range):
    """Compute the empirical distribution of 'X' over a support from a subset.

    Args:
        sub (DataFrame): Must contain columns ['X', 'W'].
        x_range (list): Ordered support values for column 'X'.

    Returns:
        np.ndarray: Distribution over x_range, or zeros if empty.
    """
    bin = len(x_range)
    distribution = np.zeros(bin)

    for i in range(bin):
        subset = sub[sub['X'] == x_range[i]]
        if subset.shape[0] > 0:
            distribution[i] = sum(subset['W'])

    return distribution / sum(distribution) if sum(distribution) > 0 else distribution
