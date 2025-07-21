import numpy as np


def normalise(tem_dist):
    """Normalise a list so it sums to 1.

    Args:
        tem_dist (Sequence[float]): Any 1-D iterable of non-negative weights.

    Returns:
        list[float]: Element-wise proportions that sum exactly to 1.
    """
    total = sum(tem_dist)
    return [val / total for val in tem_dist]

def c_generate(x_range):
    """Create an |x|*|x| L1 cost matrix for a 1-D discrete support.

    The cost between states i and j is abs(x_range[i] - x_range[j]).

    Args:
        x_range (Sequence[Number]): Ordered support points.

    Returns:
        numpy.ndarray: Square cost matrix with shape (n, n).
    """
    bin = len(x_range)
    C = np.zeros((bin, bin))

    for i in range(bin):
        for j in range(bin):
            C[i, j] = abs(x_range[i] - x_range[j])

    return C

def c_generate_higher(x_range, weight):
    """Create an L1 cost matrix for multi-dimensional discrete support.

    The distance between two such vectors is the weighted L1
    norm:
        C[i, j] = SUM over d( w[d] * |x_i[d] - x_j[d]| )

    Args:
        x_range (Sequence[Sequence[Number]]):
            Ordered support points in D dimensions.
        weight (Sequence[float]):
            Length-D vector of coordinate weights w_d.

    Returns:
        numpy.ndarray: Square cost matrix with shape (n, n).
    """   
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

