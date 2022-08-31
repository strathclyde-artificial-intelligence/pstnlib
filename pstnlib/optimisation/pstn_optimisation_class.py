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
        self.current_model = initial
        self.results = []

        if self.correlated == False:
            self.sub_problems = self.network.get_probabilistic_constraints()
        elif self.correlated == True:
            # print([c.get_description() for c in self.network.correlations])
            # print("\n")
            # print([c.get_description() for c in self.network.get_independent_probabilistic_constraints()])
            # print("\n")
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
            l = initial.addVar(name = c.get_description() + "_zl", lb = max(0.00001, c.mean - 6 * c.sd))
            u = initial.addVar(name = c.get_description() + "_zu", ub = c.mean + 6* c.sd)
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
        initial.optimize()

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

        if m.status == GRB.OPTIMAL:
            print('\n objective: ', m.objVal)
            print('\n Vars:')
            for v in m.getVars():
                if "lam" in v.varName and v.x == 0:
                    continue
                else:
                    print("Variable {}: ".format(v.varName) + str(v.x))
            m.write("junk/model.sol")
        else:
            print("\nOptimisation failed")

        return m

    def restricted_master_problem(self):
        pass

    def column_generation_problem(self, to_approximate):
        """
        Takes an instance of either probabilistic constraint or correlation and optimises reduced cost
        to find improving column
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
            dual_u = np.array([c.getAttr("Pi") for c in c_u])
            dual_l = np.array([c.getAttr("Pi") for c in c_l])
            dual_z = np.concatenate((dual_u, dual_l))
            dual_sum_lambda = c_sum_lambda.getAttr("Pi")
            
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
                return phi - np.dot(z, dual_z) - dual_sum_lambda
            
            def gradf(z):
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
            return z, f, status

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
                return phi - np.dot(z, dual_z) - dual_sum_lambda
            
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
            return z, f, status
        else:
            raise AttributeError("Invalid input type. Column generation takes instance of probabilistic constraint of correlation.")

    def optimise(self, max_iterations: int = 50, tolerance = 1e-6):
        """
        Finds schedule that optimises probability of success using column generation
        """
        start = time()
        no_iterations = 0
        # Solves restricted master problem using initial points and saves solution. 
        model = self.build_initial_model()
        self.results.append(OptimisationSolution(model, time() - start))
        self.model = model

        columns = {}
        function_values = []
        # Solves the column generation problem for each sub problem.
        for sp in self.sub_problems:
            z, f, status = self.column_generation_problem(sp)
            columns[sp.get_description()] = z
            function_values.append(f)

        # If all of the sub problems resulted in non-negative reduced cost we can terminate.
        # We define an alowable tolerance on the reduced cost which we check against.
        while all(i > -tolerance for i in function_values) == False:
            # If not we add the columns and continue.
            

        




        




        


        # # Solves column generation problem for each subproblem.
        # for problem in subproblem:
        #     do self.column_generation():
    #     #Adds uncontrollable constraints
    #     # Gets matrices for joint chance constraint P(Psi omega <= T * vars + q) >= 1 - alpha
    #     for i in range(len(cu)):
    #         incoming = PSTN.incomingContingent(cu[i])
    #         if incoming["start"] != None:
    #             incoming = incoming["start"]
    #             start_i, end_i = vars.index(incoming.source.id), vars.index(cu[i].sink.id)
    #             T[ub, start_i], T[ub, end_i] = 1, -1
    #             T[lb, start_i], T[lb, end_i] = -1, 1
    #             q[ub] = cu[i].intervals["ub"]
    #             q[lb] = -cu[i].intervals["lb"]
    #             rvar_i = rvars.index("X" + "_" + incoming.source.id + "_" + incoming.sink.id)
    #             psi[ub, rvar_i] = -1
    #             psi[lb, rvar_i] = 1
    #             mu_X[rvar_i] = incoming.mu
    #             D[rvar_i][rvar_i] = incoming.sigma
    #         elif incoming["end"] != None:
    #             incoming = incoming["end"]
    #             start_i, end_i = vars.index(cu[i].source.id), vars.index(incoming.source.id)
    #             T[ub, start_i], T[ub, end_i] = 1, -1
    #             T[lb, start_i], T[lb, end_i] = -1, 1
    #             q[ub] = cu[i].intervals["ub"]
    #             q[lb] = -cu[i].intervals["lb"]
    #             rvar_i = rvars.index("X" + "_" + incoming.source.id + "_" + incoming.sink.id)
    #             psi[ub, rvar_i] = 1
    #             psi[lb, rvar_i] = -1
    #             mu_X[rvar_i] = incoming.mu
    #             D[rvar_i][rvar_i] = incoming.sigma
    #         else:
    #             raise AttributeError("Not an uncontrollable constraint since no incoming pstc")
    #     # Gets covariance matrix from correlation matrix
    #     cov_X = D @ corr @ np.transpose(D)

    #     if correlated == False:
    #         cp = self.network.get_probabilistic_constraints()
    #         for c in cp:
    #             # Adds a variable for the lower and upper bound on the cdf.
    #             l, u = self.model.addVar(name = c.get_description() + "_l"), self.model.addVal(name = c.get_description() + "_u")
    #             # Gets all uncontrollable constraints succeeding the probabilistic constraint.
    #             outgoing = self.network.get_outgoing_uncontrollable_edge_from_timepoint(c.sink)
    #             incoming = self.network.get_incoming_uncontrollable_edge_from_timepoint(c.sink)
    #             # if constraint is the form l_ij <= bj - (b_i + X_i) <= u_ij, the uncontrollable constraint is pointing away from the uncontrollable timepoint
    #             for co in outgoing:
    #                 source, sink = c.source, co.sink
    #                 i, j = self.network.time_points.index(source), self.network.time_points.index(sink)
    #                 self.model.addConstr(-l <= x[i] - x[j] + co.ub, name = co.get_description() + "lb")
    #                 self.model.addConstr(u <= x[j] - x[i] - co.lb, name = co.get_description() + "ub")
    #             # if constraint is the form l_ij <= (b_j + X_j) - b_i -  <= u_ij, the uncontrollable constraint is pointing away from the uncontrollable timepoint
    #             for ci in incoming:
    #                 source, sink = ci.source, c.source
    #                 i, j = self.network.time_points.index(source), self.network.time_points.index(sink)
    #                 self.model.addConstr(-l <= x[j] - x[i] - ci.lb, name = ci.get_description() + "lb")
    #                 self.model.addConstr(u <= x[i] - x[j] + ci.ub, name = ci.get_description() + "ub")
                
            
    #     # Adds variable for time-points
    #     x = self.model.addMVar(len(self.network.time_points), name=[str(t.id) for t in self.network.time_points])
    # #     vars = [t.id for t in network.time_points]
    # #     rvars = ["X_{}_{}".format(c.source.id, c.sink.id) for c in network.get_probabilistic_constraints()]
    # #     cc = network.get_controllable_constraints
    # #     cu = network.get_uncontrollable_constraints

    # #     n = len(vars)
    # #     m = 2 * len(cc)

    # #     # Gets matrices for controllable constraints in the form Ax <= b
    # #     self.A = np.zeros((m, n))
    # #     self.b = np.zeros(m)
    # #     for i in range(len(cc)):
    # #         ub = 2 * i
    # #         lb = ub + 1
    # #         start_i, end_i = vars.index(cc[i].source.id), vars.index(cc[i].sink.id)
    # #         self.A[ub, start_i], self.A[ub, end_i], self.b[ub] = -1, 1, cc[i].ub
    # #         self.A[lb, start_i], self.A[lb, end_i], self.b[lb] = 1, -1, -cc[i].lb
        
    # #     # Gets the initial point from the schedule
    # #     x0 = np.array([schedule[i] for i in vars])

    # #     # For independent uncontrollable constraints:


    # # #     # Gets matrices for joint chance constraint P(Psi omega <= T * vars + q) >= 1 - alpha
    # # #     for i in range(len(cu)):
    # # #         ub = 2 * i
    # # #         lb = ub + 1
    # # #         incoming = PSTN.incomingContingent(cu[i])
    # # #         if incoming["start"] != None:
    # # #             incoming = incoming["start"]
    # # #             start_i, end_i = vars.index(incoming.source.id), vars.index(cu[i].sink.id)
    # # #             T[ub, start_i], T[ub, end_i] = 1, -1
    # # #             T[lb, start_i], T[lb, end_i] = -1, 1
    # # #             q[ub] = cu[i].intervals["ub"]
    # # #             q[lb] = -cu[i].intervals["lb"]
    # # #             rvar_i = rvars.index("X" + "_" + incoming.source.id + "_" + incoming.sink.id)
    # # #             psi[ub, rvar_i] = -1
    # # #             psi[lb, rvar_i] = 1
    # # #             mu_X[rvar_i] = incoming.mu
    # # #             D[rvar_i][rvar_i] = incoming.sigma
    # # #         elif incoming["end"] != None:
    # # #             incoming = incoming["end"]
    # # #             start_i, end_i = vars.index(cu[i].source.id), vars.index(incoming.source.id)
    # # #             T[ub, start_i], T[ub, end_i] = 1, -1
    # # #             T[lb, start_i], T[lb, end_i] = -1, 1
    # # #             q[ub] = cu[i].intervals["ub"]
    # # #             q[lb] = -cu[i].intervals["lb"]
    # # #             rvar_i = rvars.index("X" + "_" + incoming.source.id + "_" + incoming.sink.id)
    # # #             psi[ub, rvar_i] = 1
    # # #             psi[lb, rvar_i] = -1
    # # #             mu_X[rvar_i] = incoming.mu
    # # #             D[rvar_i][rvar_i] = incoming.sigma
    # # #         else:
    # # #             raise AttributeError("Not an uncontrollable constraint since no incoming pstc")
    # # #     # Gets covariance matrix from correlation matrix
    # # #     cov_X = D @ corr @ np.transpose(D)

    # # #     # Performs transformation of X into eta where eta = psi X such that eta is a p dimensional random variable
    # # #     mu_eta = psi @ mu_X
    # # #     cov_eta = psi @ cov_X @ np.transpose(psi)
    # # #     # Adds regularization term to diagonals of covariance to prevent singularity
    # # #     cov_eta = cov_eta + 1e-6*np.identity(p)
    # # #     # Translates random vector eta into standard form xi = N(0, R) where R = D.eta.D^T
    # # #     # D = np.zeros((p, p))
    # # #     # for i in range(p):
    # # #     #     D[i, i] = 1/sqrt(cov_eta[i, i])
    # # #     # R = D @ cov_eta @ D.transpose()
    # # #     # T = D @ T
    # # #     # q = D @ (q - mu_eta)
    # # #     # mu_xi = np.zeros((p))
    # # #     # cov_xi = R
    # # #     z0 = T @ x0 + q


    # # # def __init__(self, A, vars, b, c, T, q, mean, cov, psi):
    # # #     self.A = None
    # # #     self.vars = None
    # # #     self.b = None
    # # #     self.c = None
    # # #     self.T = None
    # # #     self.q = None
    # # #     self.mean = None
    # # #     self.cov = None
    # # #     self.psi = None
    # # #     self.z = None
    # # #     self.phi = None
    # # #     self.duals = None
    # # #     self.cbasis = None
    # # #     self.solved = False
    # # #     self.solution = None
    # # #     self.solution_time = None
    # # #     self.convergence_time = None
    # # #     self.master_time = None
    # # #     self.new_cols = None
    # # #     self.new_phis = None

    # # # def getDuals(self):
    # # #     # Returns a copy of the dual variable dictionary
    # # #     return copy.deepcopy(self.duals)
    
    # # # def getSolution(self):
    # # #     return self.solution
    
    # # # def addColumn(self, z_k, phi_k):
    # # #     # Adds a column z_k to matrix z and item phi_k to vector of phi values
    # # #     #try:
    # # #     self.z = np.hstack((self.z, z_k))
    # # #     self.phi = np.append(self.phi, phi_k)
    # # #     #except:
    # # #      #   raise AttributeError("Matrix z and vector phi not yet initialised")
    
    # # # def calculatePhi(self, z):
    # # #     # Calculates value of -log(F(z)) for a column z
    # # #     return -log(prob(flatten(z), self.mean, self.cov))

    # # # def reducedCost(self, z):
    # # #     # Calculates reduced cost using current dual variables and column
    # # #     return self.calculatePhi(z) - np.transpose(self.duals["mu"])@z - self.duals["nu"]
    
    # # # def setSolved(self, status):
    # # #     self.status = status
    
    # # # # Takes a Gurobi model and adds a solution containing variable and objective values.
    # # # def addSolution(self, model):
    # # #     solution = {}
    # # #     for v in model.getVars():
    # # #         solution[v.varName] = v.x
    # # #     solution["Objective"] = model.objVal
    # # #     self.solution = solution
    
    # # # def setSolutionTime(self, time):
    # # #     self.solution_time = time
    
    # # # def getCurrentProbability(self):
    # # #     #print(self.solution.keys())
    # # #     for key in self.solution.keys():
    # # #         if "phi" in key:
    # # #             return exp(-self.solution[key])
    
    # # # def add_convergence_time(self, time, gap):
    # # #     self.convergence_time.append((time, gap))
    
    # # # def add_master_time(self, time, cost):
    # # #     self.master_time.append((time, cost))
    


    
    
