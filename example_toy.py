from otpl.pddl.parser import Parser
from otpl.plans.temporal_plan import PlanTemporalNetwork
from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
from pstnlib.temporal_networks.probabilistic_temporal_network import ProbabilisticTemporalNetwork
from pstnlib.temporal_networks.temporal_network import TemporalNetwork
from pstnlib.temporal_networks.correlation import Correlation
from pstnlib.temporal_networks.timepoint import TimePoint
from pstnlib.temporal_networks.constraint import Constraint, ProbabilisticConstraint
from pstnlib.optimisation.pstn_optimisation_class import PstnOptimisation
from pstnlib.optimisation.paris import paris
from pstnlib.optimisation.solution import Solution
from random_generation import generate_random_constraints, generate_random_stn, sample_probabilistic_constraints
import numpy as np
from time import time
from graphviz import Digraph
import os
import random
inf = 1e9

# # Makes timepoints and constraints.
b0 = TimePoint(0, "Begin Travel 1")
b1 = TimePoint(1, "End Travel 1")
b2 = TimePoint(2, "Begin Travel 2")
b3 = TimePoint(3, "End Travel 2")
c1 = ProbabilisticConstraint(b0, b1, "Travel 1", {"mean": 60, "sd": 10})
c2 = Constraint(b1, b2, "Collect", {"lb": 0, "ub": inf})
c3 = ProbabilisticConstraint(b2, b3, "Travel 2", {"mean": 100, "sd": 25})
c4 = Constraint(b0, b3, "Deadline", {"lb": 0, "ub": 160})

# # # # # Makes toy stn from paper
# cstn = CorrelatedTemporalNetwork()
# cstn.time_points = [b0, b1, b2, b3]
# cstn.constraints = [c1, c2, c3, c4]
# cstn.name = "toy_example"
# cstn.plot_dot_graph()

# # Solves using column generation, assuming independence
# op = PstnOptimisation(cstn, verbose=True)
# op.optimise()
# convex = op.solutions[-1]
# convex_s = convex.get_schedule()
# test_schedule = {"0": 0.0, "2": 62.0}
# test_schedule_2 = {"0": 0.0, "2": 67.0}
# schedules = [convex_s, test_schedule, test_schedule_2]

# # Gets Monte Carlo Probability
# probs = cstn.monte_carlo(schedules)

# print("\nPROBABILITY: ")
# print("\tRMP: ", convex.get_probability())

# print("\nSchedule: ")
# print("\tRMP: ", convex_s)

# print("\nMont Carlo:")
# print("\tRMP: ", probs[0])
# print("Delta = 62: ", probs[1])
# print("Delta = 67: ", probs[2])

# # Taking correlation into consideration
cstn2 = CorrelatedTemporalNetwork()
cstn2.time_points = [b0, b1, b2, b3]
cstn2.constraints = [c1, c2, c3, c4]
corr = Correlation([c1, c3])
cstn2.add_correlation(corr)
corr.add_correlation(np.array([[1.0, 0.9], [0.9, 1.0]]))
cstn2.name = "toy_example_with_correlation"

# Solves using column generation, assuming independence
op = PstnOptimisation(cstn2, verbose=True)
op.optimise()
convex = op.solutions[-1]
convex_s = convex.get_schedule()

test_schedule = {"0": 0.0, "2": 62.0}
test_schedule_2 = {"0": 0.0, "2": 64.0}
test_schedule_3 = {"0": 0.0, "2": 67.0}
schedules = [convex_s, test_schedule, test_schedule_2, test_schedule_3]

# Gets Monte Carlo Probability
probs = cstn2.monte_carlo(schedules)
print("\nPROBABILITY: ")
print("\tRMP: ", convex.get_probability())

print("\nSchedule: ")
print("\tRMP: ", convex_s)

print("\nMont Carlo:")
print("\tRMP: ", probs[0])
print("\tDelta = 62: ", probs[1])
print("\tDekta = 64: ", probs[2])
print("\tDekta = 67: ", probs[3])
