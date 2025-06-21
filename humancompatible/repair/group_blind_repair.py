import numpy as np

from methods.coupling_utils import tmp_generator
from methods.metrics import newton


class GroupBlindRepair:
    """Group-Blind Repair (GBR) couplings for fairness-aware optimal transport.

    The class provides three variants of the original Sinkhorn-Knopp
    algorithm described in:
    https://arxiv.org/abs/2310.11407
    "Group-blind optimal transport to group parity and its constrained variants":
    and
    https://arxiv.org/abs/2410.02840
    "Overcoming Representation Bias in Fairness-Aware data Repair using Optimal Transport"
    """

    def __init__(self, C, px, ptx, V=None, epsilon=0.01, K=200):
        """Initialize a Group-Blind Repair solver.

        Args:
            C (array_like): Square cost matrix (shape [n, n]).
            px (array_like): Column vector of the source distribution.
            ptx (array_like): Column vector of the target distribution.
            V (array_like, optional): Signed imbalance vector (required for
                total and partial repairs).
            epsilon (float, optional): Entropic regularisation strength.
                Defaults to 1e-2.
            K (int, optional): Maximum number of outer Sinkhorn iterations.
                Defaults to 200.
        """
        self.C = np.asarray(C)
        self.px = np.asarray(px).ravel()[:, None]
        self.ptx = np.asarray(ptx).ravel()[:, None]
        self.V = None if V is None else np.asarray(V).ravel()[:, None]
        self.eps = epsilon
        self.K = K
        self._gamma = None  # will hold the fitted coupling
    
    def fit_baseline(self):
        """Fit the baseline (unconstrained) entropic coupling.

        Returns:
            GroupBlindRepair: The fitted instance.
        """
        self._gamma = self._baseline_core(self.C, self.eps,
                                          self.px, self.ptx, self.K)
        return self
    
    def fit_total(self):
        """Fit the total-repair coupling (strict fairness).

        Raises:
            ValueError: If no V was supplied at construction time.

        Returns:
            GroupBlindRepair: The fitted instance.
        """
        if self.V is None:
            raise ValueError("V must be provided for total repair.")
        self._gamma = self._total_core(self.C, self.eps,
                                       self.px, self.ptx, self.V, self.K)
        return self
    
    def fit_partial(self, theta_scale):
        """Fit the partial-repair coupling (fairness with slack).

        Args:
            theta_scale (float): Allowable slack theta in the constraint.

        Raises:
            ValueError: If no V was supplied at construction time.

        Returns:
            GroupBlindRepair: The fitted instance.
        """
        if self.V is None:
            raise ValueError("V must be provided for partial repair.")
        self._gamma = self._partial_core(self.C, self.eps,
                                         self.px, self.ptx,
                                         self.V, theta_scale, self.K)
        return self
    
    def coupling_matrix(self):
        """Retrieve the learned transport plan gamma.

        Returns:
            numpy.ndarray: Copy of the [n, n] coupling matrix.

        Raises:
            RuntimeError: If no fit_* method has been called yet.
        """
        if self._gamma is None:
            raise RuntimeError("Call fit_*() first.")
        return self._gamma.copy()
    

    @staticmethod
    def _baseline_core(C, e, px, ptx, K):
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

    @staticmethod
    def _total_core(C, e, px, ptx, V, K):
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

    @staticmethod
    def _partial_core(C, e, px, ptx, V, theta_scale, K):
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
