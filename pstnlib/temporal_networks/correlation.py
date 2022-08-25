from typing import Dict
from pstnlib.temporal_networks.constraint import Constraint
import numpy as np
from scipy import stats
from math import sqrt

class Correlation:
    """
    Represents a correlation across a number of probabilistic "pstc" type constraints. Given n probabilistic constraints, such that for i = 1,2,...,n,
    constraint i has mean = mu_i and standard deviation = sigma_i we have mean vector mu = (mu_1, mu_2,...,mu_n) and auxiliary matrix = [[sigma_1, 0, 0], [0, sigma_2, 0],.., [0, 0, sigma_n]]
    We add a positive definite correlation matrix R, such that the covariance matrix is Sigma = auxiliary R auxiliary^T
    """
    def __init__(self, constraints: list[Constraint]):
        self.constraints = constraints
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
        self.approximation = None

    def __str__(self) -> None:
        """
        used to print the correlation in a user-friendly way
        """
        
        return "Constraints: " +  str([c.get_description() for c in self.constraints]) + " , " + "Mean: " + str(self.mean) + " , " + "Correlation: " +  str(self.correlation)
    
    def to_json(self) -> dict:
        """
        Returns a dictionary of correlation for json printing
        """
        return {"constraints": [c.to_json() for c in self.constraints], "mean": list(self.mean), "correlation": [list(i) for i in self.correlation]}
        
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
    
    def add_random_correlation(self, eta = 0.5):
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
    
    def evaluate_probability(self, l, u):
        """
        If correlation is an n dimensional random variable X, l and u are n dimensional vectors at which to avaluate the CDF:
        returns P(l <= X <= u)
        """
        # Converts into form F(xi <= z), where z = [u, -l]^T.
        omega = np.zeros((2 * len(self.constraints), len(self.constraints)))
        for i in range(len(self.constraints)):
            omega[i, i] = 1
            omega[i + len(self.constraints), i] = -1

        xi_mean = omega @ self.mean
        xi_cov = omega @ self.covariance @ omega.transpose()

        ls = np.array([-i for i in l])
        us = np.array(u)
        z = np.concatenate([us, ls])
        return stats.multivariate_normal(xi_mean, xi_cov, allow_singular=True).cdf(z)

    def get_columns(self):
        """
        Returns matrix of columns representing generated points so far.
        """
        if self.approximation == None:
            raise AttributeError("No appoximation points to generate columns")
        else:
            l = self.approximation["points"][0][0]
            u = self.approximation["points"][0][1]
            # Makes into column vector
            l = np.c_[l]
            u = np.c_[u]
            if len(self.approximation["points"]) > 1:
                for i in range(1,len(self.approximation["points"])):
                    point = self.approximation["points"][i]
                    l_new, u_new = np.c_[np.array(point[0])], np.c_[np.array(point[1])]
                    l = np.hstack((l, l_new))
                    u = np.hstack((u, u_new))
        return (l, u)

    def get_description(self) -> str:
        """
        returns a string of the from c(source.id, sink.id)
        """
        to_return = "Corr("
        for constraint in self.constraints:
            to_add = "({},{}),".format(constraint.source.id, constraint.sink.id)
            to_return += to_add
        to_return = to_return[:-1]
        to_return += ")"
        return to_return