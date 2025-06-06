import numpy as np


def normialise(tem_dist):
    total = sum(tem_dist)
    return [val / total for val in tem_dist]

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

