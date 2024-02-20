from pstnlib.temporal_networks.constraint import Constraint
from pstnlib.optimisation.probabilities import rectangular_probability, rectangular_gradient
import numpy as np
from scipy import stats
from math import sqrt

class Correlation:
    """
    Represents a correlation across a number of probabilistic "pstc" type constraints. Given n probabilistic constraints, such that for i = 1,2,...,n,
    constraint i has mean = mu_i and standard deviation = sigma_i we have mean vector mu = (mu_1, mu_2,...,mu_n) and
    auxiliary matrix = [[sigma_1, 0, 0], [0, sigma_2, 0],.., [0, 0, sigma_n]]. We add a positive definite correlation matrix R,
    such that the covariance matrix is Sigma = auxiliary R auxiliary^T
    ----------------------
    Params:
            constraints:    list[Constraint]
                list of probabilistic constraints to include in the correlation.
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
        self.eta_used = None

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
        # If no errors, updates
        self.correlation = correlation
        self.covariance = self.auxiliary @ self.correlation @ self.auxiliary.transpose()
        # Tries to make a multivariate normal distribution. This should raise a ValueError if correlation matrix is not positive-semidefinite
        stats.multivariate_normal(self.mean, self.covariance)
    
    def add_random_correlation(self):
        """
        Description:    Code for generating random positive semidefinite correlation matrices. Taken from https://gist.github.com/junpenglao/b2467bb3ad08ea9936ddd58079412c1a
                        based on code from "Generating random correlation matrices based on vines and extended onion method", Daniel Lewandowski, Dorots Kurowicka and Harry Joe, 2009.
        Input:          eta:    Parameter - the larger eta is, the closer to the identity matrix will be the correlation matrix (more details see https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices)
        Output:         Correlation matrix
        """
        random_array = np.random.rand(len(self.constraints))
        eigs = random_array/sum(random_array)*  len(self.constraints)
        correlation = stats.random_correlation.rvs(eigs).round(decimals=4)
        self.correlation = correlation
        self.covariance = self.auxiliary @ self.correlation @ self.auxiliary.transpose()
    
    def evaluate_probability(self, l, u):
        """
        If correlation is an n dimensional random variable X, l and u are n dimensional vectors at which to avaluate the CDF:
        returns P(l <= X <= u)
        """
        if type(l) == list and type(u) == list:
            l, u = np.array(l), np.array(u)
        return rectangular_probability(self.mean, self.covariance, l, u)

    def evaluate_gradient(self, l, u):
        '''
        Calculates the gradient of the function F(u) - F(l).
        '''
        if type(l) == list and type(u) == list:
            l, u = np.array(l), np.array(u)
        return rectangular_gradient(self.mean, self.covariance, l, u)

    def add_approximation_point(self, l, u, phi):
        """
        If approximation point does not already exist in the approximation
        we add it.
        """
        for point in self.approximation["points"]:
            if (point[0]==l).all() and (point[1] == u).all():
                return False
        self.approximation["points"].append((l, u))
        self.approximation["evaluation"].append(phi)
        return True

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
    