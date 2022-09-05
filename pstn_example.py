from numpy import correlate
from otpl.pddl.parser import Parser
from otpl.plans.temporal_plan import PlanTemporalNetwork
from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
from pstnlib.temporal_networks.probabilistic_temporal_network import ProbabilisticTemporalNetwork
from pstnlib.temporal_networks.temporal_network import TemporalNetwork
from pstnlib.temporal_networks.correlation import Correlation
from pstnlib.temporal_networks.constraint import Constraint
from pstnlib.optimisation.pstn_optimisation_class import PstnOptimisation
from pstnlib.optimisation.paris import paris
from random_uncertainties import save_random_uncertainties
import random
import numpy as np

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

# # Generates random uncertainties from domain and problem and saves to folder
uncertainties = "temporal-planning-domains/rovers-metric-time-2006/uncertainties/rovers_instance-2_uncertainties_1.json"
# save_random_uncertainties(domain, problem, uncertainties)

# makes pstn, reads uncertainties from json and saves pstn as json
pstn = ProbabilisticTemporalNetwork()
pstn.parse_from_temporal_network(network)
pstn.parse_uncertainties_from_json(uncertainties)
pstn.name = "rovers_instance_2_pstn"
pstn.plot_dot_graph()
pstn.save_as_json("junk/test.json")

# loads the saved pstn from json
pstn2 = ProbabilisticTemporalNetwork()
pstn2.parse_from_json("junk/test.json")
pstn2.name = "rovers_instance_2_pstn_copy"
pstn2.plot_dot_graph()

# Solve using PARIS Algorithm.
m = paris(pstn2)
# Gets values of probability variables and computes joint outcome
prob = 1
for constraint in pstn2.get_probabilistic_constraints():
    lower, upper = m.getVarByName(constraint.get_description() + "_Fl"), m.getVarByName(constraint.get_description() + "_Fu")
    prob *= (1 - (lower.x + upper.x))

print(prob)

#Gets random probabilistic constraints to add correlation between
correlated_edges = random.sample(pstn2.get_probabilistic_constraints(), 4)
corr1, corr2 = correlated_edges[:2], correlated_edges[2:]
corr1, corr2 = Correlation(corr1), Correlation(corr2)

#Gets probabilistic constraints of choice and adds correlation between them.
corr1 = Correlation([pstn2.get_constraint_by_timepoint(pstn2.get_timepoint_by_id(3), pstn2.get_timepoint_by_id(4)), pstn2.get_constraint_by_timepoint(pstn2.get_timepoint_by_id(7), pstn2.get_timepoint_by_id(8))])
corr2 = Correlation([pstn2.get_constraint_by_timepoint(pstn2.get_timepoint_by_id(5), pstn2.get_timepoint_by_id(6)), pstn2.get_constraint_by_timepoint(pstn2.get_timepoint_by_id(9), pstn2.get_timepoint_by_id(10))])
# Adds a random psd correlation matrix
corr1.add_correlation(np.array([[1, 0.4], [0.4, 1]]))
corr2.add_correlation(np.array([[1, -0.2], [-0.2, 1]]))

# Makes a correlated temporal network
cpstn = CorrelatedTemporalNetwork()
cpstn.parse_from_probabilistic_temporal_network(pstn2)
cpstn.add_correlation(corr1)
cpstn.add_correlation(corr2)
cpstn.save_as_json("junk/cpstn")

# Testing save/load to/from json
cpstn2 = CorrelatedTemporalNetwork()
cpstn2.parse_from_json("junk/cpstn")
cpstn2.save_as_json("junk/cpstn2")

for constraint in cpstn.get_probabilistic_constraints():
    incoming = cpstn.get_incoming_edge_from_timepoint(constraint.sink)
    outgoing = cpstn.get_outgoing_edge_from_timepoint(constraint.sink)

# Optimises
op = PstnOptimisation(cpstn)
op.optimise()