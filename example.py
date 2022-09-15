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
from random_generation import generate_random_constraints, generate_random_stns, sample_probabilistic_constraints
import numpy as np
from time import time
from graphviz import Digraph
import os
import random

domain = "temporal-planning-domains/rovers-metric-time-2006/domain.pddl"
problem = "temporal-planning-domains/rovers-metric-time-2006/instances/instance-1.pddl"
planfile = "temporal-planning-domains/rovers-metric-time-2006/plans/rovers_instance-1_plan.pddl"
output_dir = "temporal-planning-domains/rovers-metric-time-2006/networks/"
uncertainties = "temporal-planning-domains/rovers-metric-time-2006/uncertainties/"
networks = generate_random_stns(domain, problem, planfile, 10, output_dir)
# for network in networks:
#     for uncert in os.listdir(uncertainties):
#         token = uncert[:-5]
#         for i in range(2, 11):
#             cstn = CorrelatedTemporalNetwork()
#             cstn.parse_from_temporal_network(network)
#             cstn.parse_uncertainties_from_json(uncertainties + uncert)
#             correlated_constraints = sample_probabilistic_constraints(cstn, 1, i)
#             for item in correlated_constraints:
#                 corr = Correlation(item)
#                 # Adds a random correlation matrix and adds correlation to the network.
#                 corr.add_random_correlation(eta = random.uniform(0, 1))
#                 cstn.add_correlation(corr)
#             cstn.name = network.name + "_" + token + "_correlationsize{}".format(i)
#             cstn.save_as_json(output_dir + cstn.name)

network_dir = "temporal-planning-domains/rovers-metric-time-2006/networks/"
result_dir = "temporal-planning-domains/rovers-metric-time-2006/results"
for file in sorted(os.listdir(network_dir)):
    if "instance-1" in file:
        print("\nSolving instance ", file)
        cstn = CorrelatedTemporalNetwork()
        cstn.parse_from_json(network_dir + file)

        # # # # Optimises using column generation
        op = PstnOptimisation(cstn, verbose=True)
        op.optimise()
        convex = op.solutions[-1]
        convex.to_json(result_dir)

        # # # # Optimises using PARIS Algorithm.
        start = time()
        m = paris(cstn)
        lp = Solution(cstn, m, time() - start)
        lp.to_json(result_dir)

# # parse PDDL domain and problem files.
# print("Parsing PDDL domain and problem file...")
# pddl_parser = Parser()
# pddl_parser.parse_file(domain)
# pddl_parser.parse_file(problem)

# # parses plan and outputs simple temporal network.
# plan = PlanTemporalNetwork(pddl_parser.domain, pddl_parser.problem)
# plan.read_from_file(planfile)
# deadline = plan.time_sorted_happenings[-1].time
# # parses simple temporal network and makes instance of temporal network
# network = TemporalNetwork()
# network.parse_from_temporal_plan_network(plan.temporal_network)
# network = generate_random_constraints(network, deadline, 7)

# # # # Imports uncertainties friom file and makes correlated temporal network.
# uncertainties = "temporal-planning-domains/rovers-metric-time-2006/uncertainties/uncertainties_1.json"
# cstn = CorrelatedTemporalNetwork()
# cstn.parse_from_temporal_network(network)
# cstn.set_controllability_of_time_points()
# cstn.parse_uncertainties_from_json(uncertainties)

# # #Gets probabilistic constraints of choice and adds correlation between them.
# corr1 = Correlation([cstn.get_constraint_by_timepoint(cstn.get_timepoint_by_id(3), cstn.get_timepoint_by_id(4)), cstn.get_constraint_by_timepoint(cstn.get_timepoint_by_id(7), cstn.get_timepoint_by_id(8))])
# corr2 = Correlation([cstn.get_constraint_by_timepoint(cstn.get_timepoint_by_id(5), cstn.get_timepoint_by_id(6)), cstn.get_constraint_by_timepoint(cstn.get_timepoint_by_id(9), cstn.get_timepoint_by_id(10))])
# corr1.add_correlation(np.array([[1, 0.5], [0.5, 1]]))
# corr2.add_correlation(np.array([[1, 0.4], [0.4, 1]]))

# cstn.add_correlation(corr1)
# cstn.add_correlation(corr2)
# cstn.name = "random_test"
# cstn.plot_dot_graph()

# # # # Optimises using column generation
# op = PstnOptimisation(cstn)
# op.optimise()
# convex = op.solutions[-1]

# # # # Optimises using PARIS Algorithm.
# start = time()
# m = paris(cstn)
# lp = Solution(cstn, m, time() - start)

# # print("\nSCHEDULES: ")
# print("LP:")
# print("\t", lp.get_schedule())
# print("CONVEX:")
# print("\t", convex.get_schedule())

# # print("\nEMPIRICAL PROBABILITY: ")
# print("LP:")
# print("\t", lp.get_probability())
# print("CONVEX:")
# print("\t", convex.get_probability())

# print("\nACTUAL PROBABILITY: ")
# print("LP:")
# print("\t", lp.monte_carlo())
# print("CONVEX:")
# print("\t", convex.monte_carlo())

# print("\RUNTIME: ")
# print("LP:")
# print("\t", lp.runtime)
# print("CONVEX:")
# print("\t", convex.runtime)