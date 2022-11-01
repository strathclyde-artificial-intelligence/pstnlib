import gurobipy as gp
from gurobipy import GRB
from pstnlib.temporal_networks.probabilistic_temporal_network import ProbabilisticTemporalNetwork
from scipy.stats import norm

def srea_lp(network: ProbabilisticTemporalNetwork, alpha: float):
    '''
    Description:    Implementation of the PARIS algorithm for strong controllability of PSTNs from "PARIS: a Polynomial-Time, Risk-Sensitive Scheduling Algorithm for Probabilistic
                    Simple Temporal Networks with Uncertainty", Santana et al. 2016.
    Input:          PSTN - Instance of PSTN to be solved
                    pres - Number of points for which to partition the function for probabilistic constraints (if pres = 50, then LHS of mode partitioned at 50 points, and RHS of mode partitioned at 50 points)
                    solver - Defines the LP solver to be used. Can be "gurobi" or "clp"
    Output:         m - A model containing all variables, constraints and objectives
    '''
    if network.name != None:
        m = gp.Model("SREA_{}".format(network.name))
    else:
        m = gp.Model("SREA")

    # Gets relevant items from PSTN
    cc = network.get_controllable_constraints()
    cu = network.get_uncontrollable_constraints()
    cp = network.get_probabilistic_constraints()
    tc = network.get_controllable_time_points()

    #Adds Problem Variables
    for t in tc:
        m.addVar(vtype=GRB.CONTINUOUS, name=str(t.id))

    for c in cp:
        m.addVar(vtype=GRB.CONTINUOUS, obj = -1, name = c.get_description() + "_deltal")
        m.addVar(vtype=GRB.CONTINUOUS, obj = -1, name = c.get_description() + "_deltau")
        m.addVar(vtype=GRB.CONTINUOUS, name = c.get_description() + "_l")
        m.addVar(vtype=GRB.CONTINUOUS, name = c.get_description() + "_u")
    m.update()

    # Adds constraints

    # For controllable constraints.
    for c in cc:
        # Collects indices of required variables in variable vector x
        start, end = m.getVarByName(str(c.source.id)), m.getVarByName(str(c.sink.id))
        # Adds constraint of the form b_j - b_i - r_u_{ij} <= y_{ij}
        m.addConstr(end - start <= c.ub)
        # Adds constraint of the form b_i - b_j -  r_l_{ij} <= -x_{ij}
        m.addConstr(end - start >= c.lb)


    # Given the alpha value, calculates the lower and upper bound to apply to the distribution such that at most alpha of the probability mass is lost.
    bounds = {}
    for c in cp:
        x = norm(c.mean, c.sd)
        l, u = x.ppf(alpha/2), x.ppf(1 - alpha/2)
        bounds[c.get_description] = (l, u)

    # For uncontrollable constraints.
    for c in cu:
        incoming = network.get_incoming_probabilistic(c)
        ## Start time-point in constraint is uncontrollable
        if incoming["start"] != None:
            incoming = incoming["start"]
            start, end = m.getVarByName(str(incoming.source.id)), m.getVarByName(str(c.sink.id))
            delta_l, delta_u = m.getVarByName(incoming.get_description() + "_deltal"), m.getVarByName(incoming.get_description() + "_deltau")
            lb, ub = bounds[incoming.get_description()][0], bounds[incoming.get_description()][1]
            l, u = m.getVarByName(incoming.get_description() + "_l"), m.getVarByName(incoming.get_description() + "_u")
            # For constraint of the form bj - bi - l_i <= y_{ij}
            m.addConstr(end - start - c.ub == lb - delta_l)
            # For constraint of the form bi - bj + u_i <= -x_{ij}
            m.addConstr(end - start - c.lb == ub + delta_u)
            # Adds constraint so that we can easily return integral bounds incorporating the heuristic improvement
            m.addConstr(l == lb - delta_l)
            m.addConstr(u = ub + delta_u)

        ## End time-point in constraint is uncontrollable
        elif incoming["end"] != None:
            incoming = incoming["end"]
            start, end = m.getVarByName(str(c.source.id)), m.getVarByName(str(incoming.source.id))
            delta_l, delta_u = m.getVarByName(incoming.get_description() + "_deltal"), m.getVarByName(incoming.get_description() + "_deltau")
            lb, ub = bounds[incoming.get_description()][0], bounds[incoming.get_description()][1]
            l, u = m.getVarByName(incoming.get_description() + "_l"), m.getVarByName(incoming.get_description() + "_u")
            # For constraint of the form b_j + u_{ij} - b_i <= y_{ij}
            m.addConstr(start - end + c.ub == ub + delta_u)        
            # For constraint of the form b_i - bj - l_{ij} <= -x_{ij}
            m.addConstr(start - end + c.lb == lb - delta_l)
            # Adds constraint so that we can easily return integral bounds incorporating the heuristic improvement
            m.addConstr(l == lb - delta_l)
            m.addConstr(u = ub + delta_u)

    m.addConstr(m.getVarByName("0") == 0)
    m.update()
    print("\nSolving: ", network.name)
    m.optimize()
    m.write("gurobi/{}.lp".format(m.getAttr("ModelName")))
    # Checks to see whether an optimal solution is found and if so it prints the solution and objective value
    if m.status == GRB.OPTIMAL:
        m.write("gurobi/{}.sol".format(m.getAttr("ModelName")))
        print('\nObjective: ', m.objVal)
        print('\nVars:')
        for v in m.getVars():
            print("Variable {}: ".format(v.varName) + str(v.x))
    return m

def srea(network: ProbabilisticTemporalNetwork, resolution: float = 0.001, range: list = [0, 0.999]):
    # If accepted resolution has been reached can return solution.
    if range[1] - range[0] <= resolution:
        model = srea_lp(network, range[1])
        # Gets result and calculates the probability
        robustness = 1
        for constraint in network.get_probabilistic_constraints():
            l = model.getVarByName(constraint.get_description() + "_l").x
            u = model.getVarByName(constraint.get_description() + "_u").x
            dist = norm(constraint.mean, constraint.sd)
            robustness *= (dist.cdf(u) - dist.cdf(l))
        return robustness, model

    alpha_l, alpha_u = range[0], range[1]
    alpha = (alpha_l + alpha_u)/2
    result = srea_lp(network, alpha)
    # If current model is optimal it tries to increase the lower bound.
    if result.status == GRB.OPTIMAL:
        return srea(network, resolution, range=[alpha, alpha_u])
    else:
        return srea(network, resolution, range=[alpha_l, alpha])
