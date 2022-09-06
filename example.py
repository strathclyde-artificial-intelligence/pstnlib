from otpl.pddl.parser import Parser
from otpl.plans.temporal_plan import PlanTemporalNetwork
from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
from pstnlib.temporal_networks.probabilistic_temporal_network import ProbabilisticTemporalNetwork
from pstnlib.temporal_networks.temporal_network import TemporalNetwork
from pstnlib.temporal_networks.correlation import Correlation
from pstnlib.temporal_networks.constraint import Constraint
from pstnlib.optimisation.pstn_optimisation_class import PstnOptimisation
from pstnlib.optimisation.paris import paris
from pstnlib.optimisation.solution import Solution
import numpy as np
from time import time

domain = "temporal-planning-domains/rovers-metric-time-2006/domain.pddl"
problem = "temporal-planning-domains/rovers-metric-time-2006/instances/instance-2.pddl"
planfile = "temporal-planning-domains/rovers-metric-time-2006/plans/rovers_instance-2_plan.pddl"

# parse PDDL domain and problem files.
print("Parsing PDDL domain and problem file...")
pddl_parser = Parser()
pddl_parser.parse_file(domain)
pddl_parser.parse_file(problem)

# parses plan and outputs simple temporal network.
plan = PlanTemporalNetwork(pddl_parser.domain, pddl_parser.problem)
plan.read_from_file(planfile)

# parses simple temporal network and makes instance of temporal network
network = TemporalNetwork()
network.parse_from_temporal_plan_network(plan.temporal_network)

# Adds a deadline to modify solvability
deadline = 47
start_id = min([i.id for i in network.time_points])
end_id = max([i.id for i in network.time_points])
network.add_constraint(Constraint(network.get_timepoint_by_id(start_id), network.get_timepoint_by_id(end_id), "Overall deadline", {"lb": 0, "ub": deadline}))
network.name = "rovers_instance-2_stn"
network.plot_dot_graph()

# Imports uncertainties friom file
uncertainties = "temporal-planning-domains/rovers-metric-time-2006/uncertainties/rovers_instance-2_uncertainties_1.json"

# makes pstn, reads uncertainties from json and saves pstn as json
pstn = ProbabilisticTemporalNetwork()
pstn.parse_from_temporal_network(network)
pstn.parse_uncertainties_from_json(uncertainties)
pstn.name = "rovers_instance_2_pstn"

#Gets probabilistic constraints of choice and adds correlation between them.
corr1 = Correlation([pstn.get_constraint_by_timepoint(pstn.get_timepoint_by_id(3), pstn.get_timepoint_by_id(4)), pstn.get_constraint_by_timepoint(pstn.get_timepoint_by_id(7), pstn.get_timepoint_by_id(8))])
corr2 = Correlation([pstn.get_constraint_by_timepoint(pstn.get_timepoint_by_id(5), pstn.get_timepoint_by_id(6)), pstn.get_constraint_by_timepoint(pstn.get_timepoint_by_id(9), pstn.get_timepoint_by_id(10))])
corr1.add_correlation(np.array([[1, 0.5], [0.5, 1]]))
corr2.add_correlation(np.array([[1, 0.4], [0.4, 1]]))

# Makes a correlated temporal network
cpstn = CorrelatedTemporalNetwork()
cpstn.parse_from_probabilistic_temporal_network(pstn)
cpstn.add_correlation(corr1)

# Optimises using column generation
op = PstnOptimisation(cpstn)
op.optimise()
convex = op.solutions[-1]

# Optimises using PARIS Algorithm.
start = time()
m = paris(pstn)
lp = Solution(pstn, m, time() - start)

print("\nSCHEDULES: ")
print("LP:")
print("\t", lp.get_schedule())
print("CONVEX:")
print("\t", convex.get_schedule())

print("\nEMPIRICAL PROBABILITY: ")
print("LP:")
print("\t", lp.get_probability())
print("CONVEX:")
print("\t", convex.get_probability())

print("\nACTUAL PROBABILITY: ")
print("LP:")
print("\t", lp.monte_carlo())
print("CONVEX:")
print("\t", convex.monte_carlo())