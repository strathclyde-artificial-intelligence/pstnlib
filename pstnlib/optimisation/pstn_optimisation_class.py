from typing import NewType
from unicodedata import name
import numpy as np
from math import log, exp
from scipy import stats
from pstnlib.temporal_networks.probabilistic_temporal_network import ProbabilisticTemporalNetwork
from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
import gurobipy as gp
from gurobipy import GRB

class pstn_optimisation(object):
    """
    Description:    Class representing Probabilistic Temporal Network (or Correlated Temporal Network) SC as an optimisation problem of the form {min phi(z) | z <= Tx + q, Ax <= b} 
    Parameters:
                A:              numpy m x n matrix of coefficients
                vars:           numpy n dimensional decision vector
                b:              numpy m dimensional vector of RHS values
                c:              numpy n dimensional vector of objective coefficients
                T:              numpy p x n matrix of coefficients
                q:              numpy p dimensional vector of RHS values
                mean:           numpy p dimensional mean vector
                cov:            numpy p x p dimensional covariance matrix
                z:              numpy p x k matrix of current approximation points
                phi:            numpy k dimensional vector of phi(z) = -log(F(z)) for approximation point (columns in z)
                duals:          dictionary {"mu": numpy array, "nu": float} of dual variables for the
                                current iteration. The dual variables are associated with the following constraints:
                                mu: Z @ lambda - T @ x <= q, nu: 1^T @ lambda = 1
                solved:         bool True if solution is optimal else False
                solution:       dictionary {variable name: value,..,objective: objective value}
                solution_time:  float time taken to reach solution
                convergence_time List of tuples (time, gap) where time is current runtime and gap is (UB - LB)/UB     
                master_time     List of tuples (time, obj) where time is current runtime and obj is current master problem objective

    """
    def __init__(self, network: ProbabilisticTemporalNetwork, schedule = None) -> None:
        """
        Parses the probablistic temporal network and populates the parameters.
        Input:  ProbabilisticTemporalNetwork - the network to be parsed
                Schedule = A valid schedule to use as an initial point. A dictionary of the form {TimepointID: scheduled_value}
        """
        self.network = network
        self.model = gp.Model("Optimisation")
        if isinstance(network, CorrelatedTemporalNetwork):
            correlated = True
        else:
            correlated = False
        self.columns = []
        self.function_evaluations = []

        # If no schedule is provided, it finds an initial column
        initial = gp.Model("initiialisation")

        # Adds Variable for timepoints
        x = initial.addMVar(len(self.network.time_points), name=[str(t.id) for t in self.network.time_points])
        initial.update()
        # Adds controllable constraints
        cc = self.network.get_controllable_constraints()
        A = np.zeros((2 * len(cc), len(self.network.time_points)))
        b = np.zeros((2 * len(cc)))
        for i in range(len(cc)):
            ub = 2 * i
            lb = ub + 1
            start_i, end_i = self.network.time_points.index(cc[i].source), self.network.time_points.index(cc[i].sink)
            A[ub, start_i], A[ub, end_i], b[ub] = -1, 1, cc[i].ub
            A[lb, start_i], A[lb, end_i], b[lb] = 1, -1, -cc[i].lb
        initial.addConstr(A @ x <= b)

        cp = self.network.get_probabilistic_constraints()
        for c in cp:
            # Adds a variable for the lower and upper bound on the cdf. Neglects probability mass outiwth 6 standard deviations of mean
            l = initial.addVar(name = c.get_description() + "_zl", lb = c.mean - 6 * c.sd)
            u = initial.addVar(name = c.get_description() + "_zu", ub = c.mean + 6* c.sd)
            initial.update()
            # Gets all uncontrollable constraints succeeding the probabilistic constraint.
            outgoing = self.network.get_outgoing_uncontrollable_edge_from_timepoint(c.sink)
            incoming = self.network.get_incoming_uncontrollable_edge_from_timepoint(c.sink)
            # if constraint is the form l_ij <= bj - (b_i + X_i) <= u_ij, the uncontrollable constraint is pointing away from the uncontrollable timepoint
            for co in outgoing:
                source, sink = c.source, co.sink
                i, j = self.network.time_points.index(source), self.network.time_points.index(sink)
                initial.addConstr(-l <= x[i] - x[j] + co.ub, name = co.get_description() + "_lb")
                initial.addConstr(u <= x[j] - x[i] - co.lb, name = co.get_description() + "_ub")
            # if constraint is the form l_ij <= (b_j + X_j) - b_i -  <= u_ij, the uncontrollable constraint is pointing away from the uncontrollable timepoint
            for ci in incoming:
                source, sink = ci.source, c.source
                i, j = self.network.time_points.index(source), self.network.time_points.index(sink)
                initial.addConstr(-l <= x[j] - x[i] - ci.lb, name = ci.get_description() + "lb")
                initial.addConstr(u <= x[i] - x[j] + ci.ub, name = ci.get_description() + "ub")
        initial.addConstr(x[0] == 0)
        initial.setObjective(sum([v for v in initial.getVars() if "_zu" in v.varName]) + sum([-v for v in initial.getVars() if "_zl" in v.varName]), GRB.MAXIMIZE)
        initial.update()
        initial.optimize()

        # Adds initial column
        zli = [-v.x for v in initial.getVars() if "_zl" in v.varName]
        zui = [v.x for v in initial.getVars() if "_zu" in v.varName]
        zui.extend(zli)
        self.columns.append(zui)
        
        # Initialises probability as 1
        probability = 1
        if correlated == True:
            for correlation in self.network.correlations:
                # From the solution extracts the lower and upper bounds.
                l, u = [], []
                for constraint in correlation.constraints:
                    li = initial.getVarByName(constraint.get_description() + "_zl").x
                    ui = initial.getVarByName(constraint.get_description() + "_zu").x
                    l.append(li)
                    u.append(ui)
                # Calculates probability for each correlation and multiplies the total probability by the value
                F = correlation.evaluate_probability(l, u)
                probability *= F
            #For each of the probabilistic constraints that are independent (i.e. not involved in a correlation.)
            for constraint in self.network.get_independent_probabilistic_constraints():
                l = initial.getVarByName(constraint.get_description() + "_zl").x
                u = initial.getVarByName(constraint.get_description() + "_zu").x
                distribution = stats.norm(constraint.mean, constraint.sd)
                # Calculates probability for each correlation and multiplies the total probability by the value
                F = distribution.cdf(u) - distribution.cdf(l)
                probability *= F
        else:
            # If not a correlated temporal network, treat everything independently.
            for constraint in self.network.get_probabilistic_constraints():
                l = initial.getVarByName(constraint.get_description() + "_zl").x
                u = initial.getVarByName(constraint.get_description() + "_zu").x
                distribution = stats.norm(constraint.mean, constraint.sd)
                # Calculates probability for each correlation and multiplies the total probability by the value
                F = distribution.cdf(u) - distribution.cdf(l)
                probability *= F
        # Adds the value of -log
        self.function_evaluations.append(-log(probability))
                        



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
    


    
    
