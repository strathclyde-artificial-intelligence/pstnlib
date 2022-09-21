from otpl.pddl.parser import Parser
from otpl.plans.temporal_plan import PlanTemporalNetwork
from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
from pstnlib.temporal_networks.probabilistic_temporal_network import ProbabilisticTemporalNetwork
from pstnlib.temporal_networks.temporal_network import TemporalNetwork
from pstnlib.temporal_networks.correlation import Correlation
from pstnlib.temporal_networks.timepoint import TimePoint
from pstnlib.temporal_networks.constraint import Constraint
from pstnlib.optimisation.pstn_optimisation_class import PstnOptimisation
from pstnlib.optimisation.paris import paris
from pstnlib.optimisation.solution import Solution
from random_generation import generate_random_constraints, generate_random_stn, sample_probabilistic_constraints
import numpy as np
from time import time
from graphviz import Digraph
import os
import random

# Generates an STN using the parameters given.
domain = "temporal-planning-domains/rovers-metric-time-2006/domain.pddl"
problem = "temporal-planning-domains/rovers-metric-time-2006/instances/instance-1.pddl"
planfile = "temporal-planning-domains/rovers-metric-time-2006/plans/rovers_instance-1_plan.pddl"
output_dir = "temporal-planning-domains/rovers-metric-time-2006/networks/"
n_constraints = 4
n_correlations = 1
size_correlation = 2
uncertainties = 4
uncertainties_f = "temporal-planning-domains/rovers-metric-time-2006/uncertainties/uncertainties_{}.json".format(uncertainties)

stn = generate_random_stn(domain, problem, planfile, n_constraints)
cstn = CorrelatedTemporalNetwork()
cstn.name = stn.name + "_network-{}".format(n_constraints)
cstn.parse_from_temporal_network(stn)
cstn.parse_uncertainties_from_json(uncertainties_f)
cstn.plot_dot_graph()
#correlated_constraints = sample_probabilistic_constraints(cstn, n_correlations, size_correlation)[0]
constraint1 = cstn.get_constraint_by_timepoint(cstn.get_timepoint_by_id(3), cstn.get_timepoint_by_id(4))
constraint2 = cstn.get_constraint_by_timepoint(cstn.get_timepoint_by_id(7), cstn.get_timepoint_by_id(8))
correlated_constraints = [constraint1, constraint2]
corr = Correlation(correlated_constraints)
corr.add_correlation(np.array([[1, 0.8],[0.8, 1]]))
print("\nCorrelation: ", corr.correlation)
print("\nCovariance: ", corr.covariance)
cstn.add_correlation(corr)

# Solves using both methods.
# Optimises using column generation
op = PstnOptimisation(cstn, verbose=True)
op.optimise()
convex = op.solutions[-1]
convex_s = convex.get_schedule()

# Optimises using PARIS Algorithm.
start = time()
m = paris(cstn)
lp = Solution(cstn, m, time() - start)
lp_s = lp.get_schedule()

# Gets Monte Carlo Probability
probs = cstn.monte_carlo([lp_s, convex_s])

print("\nPROBABILITY: ")
print("\tLP: ", lp.get_probability())
print("\tRMP: ", convex.get_probability())

print("\nSchedule: ")
print("\tLP: ", lp_s)
print("\tRMP: ", convex_s)

print("\nMont Carlo:")
print("\tLP: ", probs[0])
print("\tRMP: ", probs[1])
