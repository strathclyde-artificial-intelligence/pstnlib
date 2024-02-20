import numpy as np
import numpy as np
import numpy as np
from scipy import stats
from math import inf
from pstnlib.temporal_networks.constraint import Constraint
from pstnlib.temporal_networks.probabilistic_temporal_network import ProbabilisticTemporalNetwork
from pstnlib.temporal_networks.constraint import ProbabilisticConstraint
import gurobipy as gp
from gurobipy import GRB
inf = np.inf
env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()


def linear_probability(constraint: ProbabilisticConstraint, n: int) -> tuple[list]:
    '''
    Returns piecewise linear points representing pdf.
    -----------------
    Params:
        constraint:     pstnlib.temporal_networks.constraint.ProbabilisticConstraint
                            Constraint to be approximated
        n:              int
                            number of points to partition each region left and right of mean. 
    -----------------
    returns:
                        tuple(list, list)
    '''
    mean = constraint.mean
    sd = constraint.sd
    norm = stats.norm(mean, sd)

    # Since the PDF of omega is unbounded, in order to seperate the probability mass at set points, we must pick some arbitrarily large upper and arbitrarily small
    # lower bound. 
    cdf = 0.001
    lb = norm.ppf(cdf)
    ub = norm.ppf(1 - cdf)

    # Splits LHS of mean into n segments.
    LHS = np.linspace(lb, mean, n)
    
    # Calculates constants and gradients for each partition in LHS of mean
    partitions_l = []
    for i in range(n - 1):
        constant = 0
        for k in range(i):
            constant += norm.pdf(LHS[k+1])*(LHS[k+1] - LHS[k])
        m, c = norm.pdf(LHS[i+1]), cdf + constant - norm.pdf(LHS[i+1])*LHS[i]
        partitions_l.append([m, c])
        
    # Splits RHS of mean into n segments.
    RHS = np.linspace(mean, ub, n)

    # Calculates constants and gradients for each partition in RHS of mean
    partitions_u = []
    for j in range(0, n-1):
        constant = 0
        for k in range(j+1, n-1):
            constant += norm.pdf(RHS[k])*(RHS[k+1] - RHS[k])
        m, c = -norm.pdf(RHS[j]), cdf + constant + norm.pdf(RHS[j])*RHS[j+1]
        partitions_u.append([m, c])
        
    return (partitions_l, partitions_u)

def paris(PSTN: ProbabilisticTemporalNetwork, pres: int = 15):
    '''
    Implementation of the PARIS algorithm for strong controllability of PSTNs from "PARIS: a Polynomial-Time, Risk-Sensitive Scheduling Algorithm for Probabilistic
    Simple Temporal Networks with Uncertainty", Santana et al. 2016.
    ---------------------
    Params:
        PSTN:   pstnlib.temporal_networks.probabilistic_temporal_network.ProbabilisticTemporalNetwork.
                    Instance of PSTN to be solved
        pres:   int
                pres - Number of points for which to partition the function for probabilistic constraints (if pres = 50, then LHS of mode partitioned at 50 points, and RHS of mode partitioned at 50 points)
    -----------------
    returns:
                gurobipy.Model
    '''
    if PSTN.name != None:
        m = gp.Model("PARIS_{}".format(PSTN.name), env=env)
    else:
        m = gp.Model("PARIS", env=env)

    # Gets relevant items from PSTN
    cc = PSTN.get_controllable_constraints()
    cu = PSTN.get_uncontrollable_constraints()
    cp = PSTN.get_probabilistic_constraints()
    tc = PSTN.get_controllable_time_points()

    #Adds Problem Variables
    m.addVar(vtype=GRB.CONTINUOUS, name = "Risk")

    for t in tc:
        m.addVar(vtype=GRB.CONTINUOUS, name=str(t.id))

    for c in cp:
        m.addVar(vtype=GRB.CONTINUOUS, obj = 1, name = c.get_description() + "_Fl")
        m.addVar(vtype=GRB.CONTINUOUS, obj = 1, name = c.get_description() + "_Fu")
        m.addVar(lb=0, ub=c.mean, vtype=GRB.CONTINUOUS, name = c.get_description() + "_l")
        m.addVar(lb=c.mean, ub=inf, vtype=GRB.CONTINUOUS, name = c.get_description() + "_u")
    m.update()

    # Adds constraints
    for c in cc:
        # Collects indices of required variables in variable vector x
        start, end = m.getVarByName(str(c.source.id)), m.getVarByName(str(c.sink.id))
        # Adds constraint of the form b_j - b_i - r_u_{ij} <= y_{ij}
        m.addConstr(end - start <= c.ub)
        # Adds constraint of the form b_i - b_j -  r_l_{ij} <= -x_{ij}
        m.addConstr(end - start >= c.lb)

    for c in cu:
        incoming = PSTN.get_incoming_probabilistic(c)
        ## Start time-point in constraint is uncontrollable
        if incoming["start"] != None and incoming["end"] == None:
            incoming = incoming["start"]
            start, end = m.getVarByName(str(incoming.source.id)), m.getVarByName(str(c.sink.id))
            omega_l, omega_u = m.getVarByName(incoming.get_description() + "_l"), m.getVarByName(incoming.get_description() + "_u")
            # For constraint of the form bj - bi - l_i <= u_{ij}
            m.addConstr(end - start - omega_l <= c.ub)
            # For constraint of the form bj - bi - u_i >= l_{ij}
            m.addConstr(end - start - omega_u >= c.lb)

        ## End time-point in constraint is uncontrollable
        elif incoming["end"] != None and incoming["start"] == None:
            incoming = incoming["end"]
            start, end = m.getVarByName(str(c.source.id)), m.getVarByName(str(incoming.source.id))
            omega_l, omega_u = m.getVarByName(incoming.get_description() + "_l"), m.getVarByName(incoming.get_description() + "_u")
            # For constraint of the form b_j + u_j - b_i <= u_{ij}      
            m.addConstr(end - start + omega_u <= c.ub)        
            # For constraint of the form b_j + l_j - b_i >= l_{ij}
            m.addConstr(end - start + omega_l >= c.lb)
        # Both start and end timepoint is uncontrollable.
        else:
            incoming_start = incoming["start"]
            incoming_end = incoming["end"]
            start, end = m.getVarByName(str(incoming_start.source.id)), m.getVarByName(str(incoming_end.source.id))
            omega_l_start, omega_u_start = m.getVarByName(incoming_start.get_description() + "_l"), m.getVarByName(incoming_start.get_description() + "_u")
            omega_l_end, omega_u_end = m.getVarByName(incoming_end.get_description() + "_l"), m.getVarByName(incoming_end.get_description() + "_u")
            # For constraint of the form b_j + u_j - (b_i + l_i) <= u_{ij}      
            m.addConstr(end + omega_u_end - start - omega_l_start <= c.ub)        
            # For constraint of the form b_i + l_j - (b_i + u_i) >= l_{ij}
            m.addConstr(end + omega_l_end - start - omega_u_start >= c.lb)

    # Adds piecewise linear approximation of probabilistic constraints.
    for c in cp:
        F_l, F_u = m.getVarByName(c.get_description() + "_Fl"), m.getVarByName(c.get_description() + "_Fu")
        omega_l, omega_u = m.getVarByName(c.get_description() + "_l"), m.getVarByName(c.get_description() + "_u")
        # Adds piecewise linear constraints.
        partitions = linear_probability(c, pres)
        partitions_l, partitions_u = partitions[0], partitions[1]
        # Adds constraints of the form F_l >= grad * l + intercept
        for partition in partitions_l:
            grad, const = partition[0], partition[1]
            m.addConstr(F_l - grad*omega_l >= const)
        for partition in partitions_u:
            grad, const = partition[0], partition[1]
            m.addConstr(F_u - grad*omega_u >= const)

    m.update()
    risk = m.getVarByName("Risk")
    m.addConstr(gp.quicksum([v for v in m.getVars() if v.varName[-2:] in ["Fu", "Fl"]]) == risk, 'risk')
    m.update()
    m.optimize()
    return m
        
        
    