from typing import NewType, runtime_checkable
from unicodedata import name
import numpy as np
from math import log, exp
from scipy import stats
from time import time
from pstnlib.temporal_networks.constraint import ProbabilisticConstraint
from pstnlib.temporal_networks.correlation import Correlation
from pstnlib.temporal_networks.probabilistic_temporal_network import ProbabilisticTemporalNetwork
from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
import gurobipy as gp
from gurobipy import GRB
from scipy import optimize
inf = 1000000000
eps = 1e-5
class OptimisationSolution(object):
    """
    Description:    Class representing a solution to the restricted master problem on a given iteration.
    Parameters:     model - Gurobi model instance containing solution.
                    runtime - cumulative runtime of current iteration.
    """
    def __init__(self, model, runtime) -> None:
        self.model = model
        self.runtime = runtime
    
    def get_schedule(self):
        """
        returns the schedule from the gurobi model solution.
        """
        pass

    def get_probability(self):
        """
        returns the probability of success from the gurobi model solution.
        """
        pass

class PstnOptimisation(object):
    """
    Description:    Class representing Probabilistic Temporal Network (or Correlated Temporal Network) SC as an optimisation problem of the form {min phi(z) | z <= Tx + q, Ax <= b} 
    Parameters:     network - Instance of Probabilisitc Temporal Network or (or Correlated Temporal Network) to be optimised.
                    results - List of instances of optimisation_colution class for each iteration

    """
    def __init__(self, network: ProbabilisticTemporalNetwork, schedule = None) -> None:
        """
        Parses the probablistic temporal network and generates initial approximation points.
        """
        self.network = network
        if isinstance(network, CorrelatedTemporalNetwork):
            self.correlated = True
        else:
            self.correlated = False
        
        # If no schedule is provided, it finds an initial column
        initial = gp.Model("initiialisation")
        self.model = initial
        self.results = []

        if self.correlated == False:
            self.sub_problems = self.network.get_probabilistic_constraints()
        elif self.correlated == True:
            self.sub_problems = self.network.get_independent_probabilistic_constraints() + self.network.correlations

        # Adds Variable for timepoints
        tc = self.network.get_controllable_time_points()
        x = initial.addMVar(len(tc), name=[str(t.id) for t in tc])
        initial.update()
        
        # Adds controllable constraints
        cc = self.network.get_controllable_constraints()
        A = np.zeros((2 * len(cc), len(tc)))
        b = np.zeros((2 * len(cc)))
        for i in range(len(cc)):
            ub = 2 * i
            lb = ub + 1
            start_i, end_i = tc.index(cc[i].source), tc.index(cc[i].sink)
            A[ub, start_i], A[ub, end_i], b[ub] = -1, 1, cc[i].ub
            A[lb, start_i], A[lb, end_i], b[lb] = 1, -1, -cc[i].lb
        initial.addConstr(A @ x <= b)

        cp = self.network.get_probabilistic_constraints()
        for c in cp:
            # Adds a variable for the lower and upper bound on the cdf. Neglects probability mass outiwth 6 standard deviations of mean
            l = initial.addVar(name = c.get_description() + "_zl", lb = max(eps, c.mean - 6 * c.sd))
            u = initial.addVar(name = c.get_description() + "_zu", ub = c.mean + 6* c.sd)
            initial.addConstr(l + eps <= u, name = c.get_description() + "l_u")
            initial.update()
            # Gets all uncontrollable constraints succeeding the probabilistic constraint.
            outgoing = self.network.get_outgoing_uncontrollable_edge_from_timepoint(c.sink)
            incoming = self.network.get_incoming_uncontrollable_edge_from_timepoint(c.sink)
            # if constraint is the form l_ij <= bj - (b_i + X_i) <= u_ij, the uncontrollable constraint is pointing away from the uncontrollable timepoint
            for co in outgoing:
                source, sink = c.source, co.sink
                i, j = tc.index(source), tc.index(sink)
                initial.addConstr(-l <= x[i] - x[j] + co.ub, name = co.get_description() + "_lb")
                initial.addConstr(u <= x[j] - x[i] - co.lb, name = co.get_description() + "_ub")
            # if constraint is the form l_ij <= (b_j + X_j) - b_i -  <= u_ij, the uncontrollable constraint is pointing away from the uncontrollable timepoint
            for ci in incoming:
                source, sink = ci.source, c.source
                i, j = tc.index(source), tc.index(sink)
                initial.addConstr(-l <= x[j] - x[i] - ci.lb, name = ci.get_description() + "lb")
                initial.addConstr(u <= x[i] - x[j] + ci.ub, name = ci.get_description() + "ub")
        initial.addConstr(x[0] == 0)
        initial.setObjective(sum([v for v in initial.getVars() if "_zu" in v.varName]) + sum([-v for v in initial.getVars() if "_zl" in v.varName]), GRB.MAXIMIZE)
        initial.update()
        initial.write("junk/initial.lp")
        initial.optimize()
        initial.write("junk/initial.sol")

        # Adds initial approximation points
        for c in self.sub_problems:
            if isinstance(c, Correlation):
                # Initialises the inner approximation for each correlation
                c.approximation = {"points": [], "evaluation": []}
                # From the solution extracts the lower and upper bounds.
                l, u = [], []
                for constraint in c.constraints:
                    li = initial.getVarByName(constraint.get_description() + "_zl").x
                    ui = initial.getVarByName(constraint.get_description() + "_zu").x
                    l.append(li)
                    u.append(ui)
                c.approximation["points"].append((l, u))
                # Calculates probability and saves the value of -log(F(z))
                F = c.evaluate_probability(l, u)
                c.approximation["evaluation"].append(-log(F))
            #For each of the probabilistic constraints that are independent (i.e. not involved in a correlation.)
            elif isinstance(c, ProbabilisticConstraint):
                # Initialises the inner approximation
                c.approximation = {"points": [], "evaluation": []}
                l = initial.getVarByName(c.get_description() + "_zl").x
                u = initial.getVarByName(c.get_description() + "_zu").x
                # Adds approximation point
                c.approximation["points"].append((l, u))
                # Calculates probability and adds funtion evaluation
                F = c.evaluate_probability(l, u)
                c.approximation["evaluation"].append(-log(F))
    
    def build_initial_model(self):
        """
        Builds and solves the restricted master problem given the initial approximation points.
        """
        m = gp.Model("RMP Iteration 0")

        # Adds Variable for timepoints
        tc = self.network.get_controllable_time_points()
        x = m.addMVar(len(tc), name=[str(t.id) for t in tc])
        m.update()

        # Adds controllable constraints
        cc = self.network.get_controllable_constraints()
        A = np.zeros((2 * len(cc), len(tc)))
        b = np.zeros((2 * len(cc)))
        for i in range(len(cc)):
            ub = 2 * i
            lb = ub + 1
            start_i, end_i = tc.index(cc[i].source), tc.index(cc[i].sink)
            A[ub, start_i], A[ub, end_i], b[ub] = -1, 1, cc[i].ub
            A[lb, start_i], A[lb, end_i], b[lb] = 1, -1, -cc[i].lb
        m.addConstr(A @ x <= b)

        for c in self.sub_problems:
            # For independent probabliistic constraints
            if isinstance(c, ProbabilisticConstraint):
                # Adds inner approximation variables.
                m.addVar(name = c.get_description() + "_lam_0", obj=c.approximation["evaluation"][0])
                m.update()
                m.addConstr(gp.quicksum([v for v in m.getVars() if c.get_description() + "_lam" in v.varName]) == 1, name=c.get_description() + "_sum_lam")
                # Gets columns
                points = c.approximation["points"]
                l, u = [p[0] for p in points], [p[1] for p in points]
                # Gets all uncontrollable constraints succeeding the probabilistic constraint.
                outgoing = self.network.get_outgoing_uncontrollable_edge_from_timepoint(c.sink)
                incoming = self.network.get_incoming_uncontrollable_edge_from_timepoint(c.sink)
                # if constraint is the form l_ij <= bj - (b_i + X_i) <= u_ij, the uncontrollable constraint is pointing away from the uncontrollable timepoint
                for co in outgoing:
                    source, sink = c.source, co.sink
                    i, j = tc.index(source), tc.index(sink)
                    m.addConstr(gp.quicksum([u[k] * m.getVarByName(c.get_description() + "_lam_{}".format(k)) for k in range(len(u))]) <= x[j] - x[i] - co.lb,  name = c.get_description() + "_" + co.get_description() + "_ub")
                    m.addConstr(gp.quicksum([-l[k] * m.getVarByName(c.get_description() + "_lam_{}".format(k)) for k in range(len(l))]) <= x[i] - x[j] + co.ub, name = c.get_description() + "_" + co.get_description() + "_lb")
                # if constraint is the form l_ij <= (b_j + X_j) - b_i -  <= u_ij, the uncontrollable constraint is pointing away from the uncontrollable timepoint
                for ci in incoming:
                    source, sink = ci.source, c.source
                    i, j = tc.index(source), tc.index(sink)
                    m.addConstr(gp.quicksum([u[k] * m.getVarByName(c.get_description() + "_lam_{}".format(k)) for k in range(len(u))]) <= x[i] - x[j] + ci.ub, name = c.get_description() + "_" + ci.get_description() + "_ub")
                    m.addConstr(gp.quicksum([-l[k] * m.getVarByName(c.get_description() + "_lam_{}".format(k)) for k in range(len(l))]) <= x[j] - x[i] - ci.lb, name = c.get_description() + "_" + ci.get_description() + "_lb")
            # For correlations
            elif isinstance(c, Correlation):
                # Adds inner approximation variables
                m.addVar(name = c.get_description() + "_lam_0", obj=c.approximation["evaluation"][0])
                m.update()
                m.addConstr(gp.quicksum([v for v in m.getVars() if c.get_description() + "_lam" in v.varName]) == 1, name=c.get_description() + "_sum_lam")
                # Gets columns
                points = c.approximation["points"]
                l, u = [p[0] for p in points], [p[1] for p in points]
                for n in range(len(c.constraints)):
                    # Gets all uncontrollable constraints succeeding the probabilistic constraint.
                    outgoing = self.network.get_outgoing_uncontrollable_edge_from_timepoint(c.constraints[n].sink)
                    incoming = self.network.get_incoming_uncontrollable_edge_from_timepoint(c.constraints[n].sink)
                    for co in outgoing:
                        source, sink = c.constraints[n].source, co.sink
                        i, j = tc.index(source), tc.index(sink)
                        m.addConstr(gp.quicksum([u[k][n] * m.getVarByName(c.get_description() + "_lam_{}".format(k)) for k in range(len(u))]) <= x[j] - x[i] - co.lb,  name = c.get_description() + "_" + c.constraints[n].get_description() + "_" + co.get_description() + "_ub")
                        m.addConstr(gp.quicksum([-l[k][n] * m.getVarByName(c.get_description() + "_lam_{}".format(k)) for k in range(len(l))]) <= x[i] - x[j] + co.ub, name = c.get_description() + "_" + c.constraints[n].get_description() + "_" + co.get_description() + "_lb")
                    for ci in incoming:
                        source, sink = ci.source, c.constraints[n].source
                        i, j = tc.index(source), tc.index(sink)
                        m.addConstr(gp.quicksum([u[k][n] * m.getVarByName(c.get_description() + "_lam_{}".format(k)) for k in range(len(u))]) <= x[i] - x[j] + ci.ub, name = c.get_description() + "_" + c.constraints[n].get_description() + "_" + ci.get_description() + "_ub")
                        m.addConstr(gp.quicksum([-l[k][n] * m.getVarByName(c.get_description() + "_lam_{}".format(k)) for k in range(len(l))]) <= x[j] - x[i] - ci.lb, name = c.get_description() + "_" + c.constraints[n].get_description() + "_" + ci.get_description() + "_lb")
        # Constrains initial time-point to be zero
        m.addConstr(x[0] == 0)
        m.update()
        m.write("junk/model.lp")
        m.optimize()
        return m

    def column_generation_problem(self, to_approximate):
        """
        Takes an instance of either probabilistic constraint or correlation and optimises reduced cost
        to find improving column. Adds all columns with negative reduced cost.
        """
        if isinstance(to_approximate, ProbabilisticConstraint):
            print("\n", to_approximate.get_description())
            distribution = stats.norm(to_approximate.mean, to_approximate.sd)
            # Gets starting point from current solution.
            points = to_approximate.approximation["points"]
            l, u = [p[0] for p in points], [p[1] for p in points]
            u0 = sum([u[k] * self.model.getVarByName(to_approximate.get_description() + "_lam_{}".format(k)).x for k in range(len(u))])
            l0 = sum([l[k] * self.model.getVarByName(to_approximate.get_description() + "_lam_{}".format(k)).x for k in range(len(l))])

            # Gets a list of constraints which use lower and upper bound of the probabilistic constraint
            c_u = [c for c in self.model.getConstrs() if to_approximate.get_description() in c.getAttr("ConstrName") and "_ub" in c.getAttr("ConstrName")]
            c_l = [c for c in self.model.getConstrs() if to_approximate.get_description() in c.getAttr("ConstrName") and "_lb" in c.getAttr("ConstrName")]
            # Gets constraint for sum of lambdas.
            c_sum_lambda = self.model.getConstrByName(to_approximate.get_description() + "_sum_lam")
            # Gets the dual values associated with each constraint.
            dual_u = np.array(sum([c.getAttr("Pi") for c in c_u]))
            dual_l = -np.array(sum([c.getAttr("Pi") for c in c_l]))
            dual_z = np.array([dual_u, dual_l])
            dual_sum_lambda = c_sum_lambda.getAttr("Pi")
            
            #################################################################
            #   THIS SHOULD OPTIMISE GIVEN L AND U THEN CONVERT AT THE END  #
            #################################################################
           
            # Makes initial vector z given initial l and u. 
            assert len(c_u) == len(c_l), "Should be same number of upper bound constraints and lower bound constraints."

            no_of_constraints = len(c_u) 
            uz0, lz0 = [], []
            for i in range(no_of_constraints):
                uz0.append(u0)
                lz0.append(-l0)
            z0 = np.concatenate((np.array(uz0), np.array(lz0)))

            def dualf(z):
                n = len(z)
                zu, zl = z[:n//2], z[n//2:]
                assert np.all(zu == zu[0])
                assert np.all(zl == zl[0])
                u, l = zu[0], -zl[0]
                phi = -log(distribution.cdf(u) - distribution.cdf(l))
                dual = phi - np.dot(z, dual_z) - dual_sum_lambda
                # If reduced cost is less than zero we can add the column.
                if dual <= 0:
                    to_approximate.approximation["points"].append((l, u))
                    to_approximate.approximation["evaluation"].append(phi)
                    # Add to gurobi model.
                    constraints = c_u + c_l + [c_sum_lambda]
                    coefficients = np.append(z, 1)
                    self.model.addVar(lb = 0, ub = 1, obj = phi, column = gp.Column(coefficients, constraints), name = to_approximate.get_description() + "_lam_{}".format(len(to_approximate.approximation["evaluation"])-1))
                return dual
            
            def gradf(z):
                if to_approximate.get_description() == "c(7,8)":
                    print("z", z)
                n = len(z)
                zu, zl = z[:n//2], z[n//2:]
                assert np.all(zu == zu[0])
                assert np.all(zl == zl[0])
                u, l = zu[0], -zl[0]
                dF = np.zeros(n)
                for i in range(n//2):
                    dF[i] = distribution.pdf(u)
                    dF[n//2 + i] = -distribution.pdf(l)
                F = distribution.cdf(u) - distribution.cdf(l)
                if to_approximate.get_description() == "c(7,8)":
                    print("grad", -dF/F - dual_z)
                return -dF/F - dual_z

            # # Adds bounds to prevent variables being non-negative
            bounds = []
            for i in range(len(z0)):
                bound = (0.00001, inf)
                bounds.append(bound)

            res = optimize.minimize(dualf, z0, jac = gradf, method = "L-BFGS-B", bounds = bounds)
            z = res.x
            f = res.fun
            status = res.success
            return f, status

        elif isinstance(to_approximate, Correlation):
            print("\n", to_approximate.get_description())
            distribution = stats.multivariate_normal(to_approximate.mean, to_approximate.covariance)
            # Gets starting point from current solution.
            points = to_approximate.approximation["points"]
            l, u = [p[0] for p in points], [p[1] for p in points]
            u0 = np.zeros(len(u[0]))
            l0 = np.zeros(len(u[0]))
            assert len(u0) == len(l0)
            for k in range(len(u)):
                u0 += np.array([u[k][i] * self.model.getVarByName(to_approximate.get_description() + "_lam_{}".format(k)).x for i in range(len(u[k]))])
                l0 += np.array([l[k][i] * self.model.getVarByName(to_approximate.get_description() + "_lam_{}".format(k)).x for i in range(len(l[k]))])

            # Gets a list of constraints which use lower and upper bound of the probabilistic constraint
            c_u = [c for c in self.model.getConstrs() if to_approximate.get_description() in c.getAttr("ConstrName") and "_ub" in c.getAttr("ConstrName")]
            c_l = [c for c in self.model.getConstrs() if to_approximate.get_description() in c.getAttr("ConstrName") and "_lb" in c.getAttr("ConstrName")]

            # Gets constraint for sum of lambdas.
            c_sum_lambda = self.model.getConstrByName(to_approximate.get_description() + "_sum_lam")
            # Gets the dual values associated with each constraint.
            dual_u = np.array([c.getAttr("Pi") for c in c_u])
            dual_l = np.array([c.getAttr("Pi") for c in c_l])
            dual_z = np.concatenate((dual_u, dual_l))
            dual_sum_lambda = c_sum_lambda.getAttr("Pi")

            # Makes initial vector z given initial l and u.
            z0 = np.zeros(2 * len(c_u))
            indexes = np.zeros(len(c_u))
            for i in range(len(c_u)):
                probabilistic = c_u[i].getAttr("ConstrName").split("_")[1]
                for j in range(len(to_approximate.constraints)):
                    if to_approximate.constraints[j].get_description() == probabilistic:
                        z0[i] = u0[j]
                        z0[i + len(c_u)] = -l0[j]
                        indexes[i] = j

            def dualf(z):
                n = len(z)
                zu, zl = z[:n//2], z[n//2:]
                # Since z can be in form z = [u1, u1, u2, -l1, -l1, -l2], we must extract in form l = [l1, l2] and u = [u1, u2]
                # Variable indexes relates the position in vector zu and zl to the index of constraint i in correlation.contraints.
                u, l = np.zeros(len(to_approximate.constraints)), np.zeros(len(to_approximate.constraints))
                for i in range(len(to_approximate.constraints)):
                    for j in range(len(indexes)):
                        if indexes[j] == i:
                            u[i] = zu[j]
                            l[i] = -zl[j]
                            break
                phi = -log(to_approximate.evaluate_probability(l, u))
                dual = phi - np.dot(z, dual_z) - dual_sum_lambda
                # If reduced cost is less than zero we can add the column.
                if dual <= 0:
                    to_approximate.approximation["points"].append((l, u))
                    to_approximate.approximation["evaluation"].append(phi)
                    # Add to gurobi model.
                    constraints = c_u + c_l + [c_sum_lambda]
                    coefficients = np.append(z, 1)
                    self.model.addVar(lb = 0, ub = 1, obj = phi, column = gp.Column(coefficients, constraints), name = to_approximate.get_description() + "_lam_{}".format(len(to_approximate.approximation["evaluation"])-1))
                return dual
            
            def gradf(z):
                n = len(z)
                zu, zl = z[:n//2], z[n//2:]
                # Since z can be in form z = [u1, u1, u2, -l1, -l1, -l2], we must extract in form l = [l1, l2] and u = [u1, u2]
                # Variable indexes relates the position in vector zu and zl to the index of constraint i in correlation.contraints.
                u, l = np.zeros(len(to_approximate.constraints)), np.zeros(len(to_approximate.constraints))
                for i in range(len(to_approximate.constraints)):
                    for j in range(len(indexes)):
                        if indexes[j] == i:
                            u[i] = zu[j]
                            l[i] = -zl[j]
                            break
                dl, du = to_approximate.evaluate_gradient(l, u)
                # As above but in reverse this adds the gradient into the vector at the appropriate position based on "indexes"
                dF = np.zeros(n)
                for i in range(len(to_approximate.constraints)):
                    for j in range(len(indexes)):
                        if indexes[j] == i:
                            dF[j] = du[i]
                            dF[j + n//2] = dl[i]
                F = to_approximate.evaluate_probability(l, u)
                return -dF/F - dual_z

            # Adds bounds to prevent variables being non-negative
            bounds = []
            for i in range(len(z0)):
                bound = (0.00001, inf)
                bounds.append(bound)

            # Finds the column z that minimizes the dual.
            res = optimize.minimize(dualf, z0, jac = gradf, method = "L-BFGS-B", bounds = bounds)
            z = res.x
            f = res.fun
            status = res.success
            return f, status
        else:
            raise AttributeError("Invalid input type. Column generation takes instance of probabilistic constraint of correlation.")

    def optimise(self, max_iterations: int = 2, tolerance = 1e-6):
        """
        Finds schedule that optimises probability of success using column generation
        """
        start = time()
        # Solves restricted master problem using initial points and saves solution. 
        self.model = self.build_initial_model()
        self.results.append(OptimisationSolution(self.model, time() - start))
        no_iterations = 1

        function_values = []
        # Solves the column generation problem for each sub problem.
        for sp in self.sub_problems:
            f, status = self.column_generation_problem(sp)
            function_values.append(f)

        # If all of the sub problems resulted in non-negative reduced cost we can terminate.
        # We define an alowable tolerance on the reduced cost which we check against.
        while all(i > -tolerance for i in function_values) == False and no_iterations < max_iterations:
            no_iterations += 1
            # If not satisfied we can run the master problem with the new columns added
            self.model.update()
            self.model.write("junk/model{}.lp".format(no_iterations))
            self.model.optimize()
            self.model.write("junk/model{}.sol".format(no_iterations))
            self.results.append(OptimisationSolution(self.model, time() - start))

            function_values = []
            # Solves the column generation problem for each sub problem.
            for sp in self.sub_problems:
                f, status = self.column_generation_problem(sp)
                function_values.append(f)
        
        if all(i > -tolerance for i in function_values) == True and self.model.status == GRB.OPTIMAL:
            print("Optimisation terminated sucessfully")
            print('\n Objective: ', self.model.objVal)
            print("Probability: ", exp(-self.model.objVal))
            print('\n Vars:')
            for v in self.model.getVars():
                if "_lam_" in v.varName and v.x == 0:
                    continue
                else:
                    print("Variable {}: ".format(v.varName) + str(v.x))
            return True
        else:
            print("\nOptimisation failed")
            print('\n Objective: ', self.model.objVal)
            print("Probability: ", exp(-self.model.objVal))
            print('\n Vars:')
            for v in self.model.getVars():
                if "_lam_" in v.varName and v.x == 0:
                    continue
                else:
                    print("Variable {}: ".format(v.varName) + str(v.x))
            return False


    
    
