from pstnlib.temporal_networks.constraint import Constraint
import numpy as np
from scipy import stats

class Correlation:
    """
    Represents a correlation across a number of probabilistic "pstc" type constraints. Given n probabilistic constraints, such that for i = 1,2,...,n,
    constraint i has mean = mu_i and standard deviation = sigma_i we have mean vector mu = (mu_1, mu_2,...,mu_n) and auxiliary matrix = [[sigma_1, 0, 0], [0, sigma_2, 0],.., [0, 0, sigma_n]]
    We add a positive definite correlation matrix R, such that the covariance matrix is Sigma = auxiliary R auxiliary^T
    """
    def __init__(self, constraints: list[Constraint]):
        self.contraints = constraints
        for c in self.constraints:
            if c.type != "pstc":
                raise AttributeError("Correlated constraints must be of type pstc (probabilistic simple temporal constraint)")
        # Initialises the correlation matrix to be an identity matrix of size n
        self.correlation = np.identity(len(self.constraints))
        self.mean = np.array([c.mean for c in self.constraints])
        # Initialises covariance matrix
        self.auxiliary = np.zeros((len(constraints), len(constraints)))
        for i in range(len(constraints)):
            for j in range(len(constraints)):
                if i == j:
                    self.auxiliary[i, j] = constraints[i].sd
        self.covariance = self.auxiliary @ self.correlation @ self.auxiliary.transpose()

    def add_correlation(self, correlation: np.ndarray) -> None:
        """
        Updates correlation matrix and covariance matrix
        """
        # Checks dimensions of correlation matrix are correct
        assert np.shape(correlation)[0] == len(self.constraints) and np.shape(correlation)[1] == len(self.constraints), "Dimensions of correlation matrix are inconsistent with number of constraints. If n is number of constraints, correlation should be n x n array."
        # Tries to make a multivariate normal distribution. This should raise a ValueError if correlation matrix is not positive-semidefinite
        stats.multivariate_normal(self.mean, correlation)
        # If no errors, updates
        self.correlation = correlation
        self.covariance = self.auxiliary @ self.correlation @ self.auxiliary.transpose()
    
    def add_random_correlation(self, eta):
        """
        Description:    Code for generating random positive semidefinite correlation matrices. Taken from https://gist.github.com/junpenglao/b2467bb3ad08ea9936ddd58079412c1a
                        based on code from "Generating random correlation matrices based on vines and extended onion method", Daniel Lewandowski, Dorots Kurowicka and Harry Joe, 2009.
        Input:          eta:    Parameter - the larger eta is, the closer to the identity matrix will be the correlation matrix (more details see https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices)
        Output:         Correlation matrix
        """
        size = 1
        n = len(self.constraints)
        beta0 = eta - 1 + n/2
        shape = n * (n-1) // 2
        triu_ind = np.triu_indices(n, 1)
        beta_ = np.array([beta0 - k/2 for k in triu_ind[0]])
        # partial correlations sampled from beta dist.
        P = np.ones((n, n) + (size,))
        P[triu_ind] = stats.beta.rvs(a=beta_, b=beta_, size=(size,) + (shape,)).T
        # scale partial correlation matrix to [-1, 1]
        P = (P-.5)*2
    
        for k, i in zip(triu_ind[0], triu_ind[1]):
            p = P[k, i]
            for l in range(k-1, -1, -1):  # convert partial correlation to raw correlation
                p = p * np.sqrt((1 - P[l, i]**2) *
                                (1 - P[l, k]**2)) + P[l, i] * P[l, k]
            P[k, i] = p
            P[i, k] = p
        self.correlation = np.transpose(P, (2, 0 ,1))[0]
        self.covariance = self.auxiliary @ self.correlation @ self.auxiliary.transpose()