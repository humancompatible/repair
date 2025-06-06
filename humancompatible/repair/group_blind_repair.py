import numpy as np
import pandas as pd

from .coupling_utils import tmp_generator


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

def newton(fun, dfun, a, stepmax, tol):
    if abs(fun(a)) <= tol:
        return a
    
    for _ in range(1, stepmax + 1):
        b = a - fun(a) / dfun(a)
        if abs(fun(b)) <= tol:
            return b
        a = b
    
    return b 
