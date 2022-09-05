from pstnlib.temporal_networks.constraint import ProbabilisticConstraint
import numpy as np
from scipy import stats
from math import log
inf = 1000000000
eps = 1e-6

class ColumnGenerationProbabilistic():
    """
    Class for minimising reduced cost for an independent probabilistic constraints. This is for compatability with cyipopt:
    https://cyipopt.readthedocs.io/en/stable/tutorial.html#scipy-compatible-interface
    """
    def __init__(self, to_solve: ProbabilisticConstraint) -> None:
        self.to_solve = to_solve
        self.distribution = stats.norm(to_solve.mean, to_solve.sd)

    def objective(self, x):
        """
        Returns the scalar value of the reduced cost given a column x = [l, u]
        """
        print("Evaluating function at point ", x)
        u, l = x[0], x[1]
        duals = self.to_solve.approximation["duals"]
        dual_u, dual_l, dual_sum_lambda = duals[0], duals[1], duals[2]
        phi = -log(self.distribution.cdf(u) - self.distribution.cdf(l) + 1e-6)
        red_cost = phi - np.dot(u, dual_u) - np.dot(l, dual_l) - dual_sum_lambda
        # If reduced cost is less than zero we can add the column.
        if red_cost <= 0:
            toAdd = self.to_solve.add_approximation_point(l, u, phi)
        print("F: ", red_cost)
        return red_cost

    def gradient(self, x):
        """
        Returns the gradient vector at a given x
        """
        print("Evaluating gradient at point ", x)
        u, l = x[0], x[1]
        duals = self.to_solve.approximation["duals"]
        dual_x = np.array([duals[0], duals[1]])
        dF = np.array([self.distribution.pdf(u), -self.distribution.pdf(l)])        
        F = self.distribution.cdf(u) - self.distribution.cdf(l) + 1e-6
        print("Probability: ", F)
        print("Grad: ", -dF/F - dual_x)
        return -dF/F - dual_x
    

