import argparse
from otpl.pddl.parser import Parser
from otpl.pddl.grounding import Grounding
from otpl.plans.temporal_plan import PlanTemporalNetwork
from otpl.temporal_networks.temporal_network import SimpleTemporalNetwork

domain = "problems/rovers/domain.pddl"
problem = "problems/rovers/instances/instance-1.pddl"
planfile = "problems/rovers/results/rovers_instance-1_plan.pddl"
uncertainties = "problems/rovers/uncertainties.json"

# parse PDDL domain and problem files
print("Parsing PDDL domain and problem file...")
pddl_parser = Parser()
pddl_parser.parse_file(domain)
pddl_parser.parse_file(problem)
    
print("Parsing PDDL plan file...")
plan = PlanTemporalNetwork(pddl_parser.domain, pddl_parser.problem)
plan.read_from_file(planfile)
plan.temporal_network.print_dot_graph()

stn = TemporalNetwork()
stn.parse_from_temporal_plan_network(plan.temporal_network)
stn.name = "rovers_instance_1_stn"
stn.plot_dot_graph()

# makes pstn, reads uncertainties from json and prints
# pstn = stn.make_pstn()
# pstn.read_uncertainties_from_json(uncertainties)
# pstn.name = "rovers_instance_1_pstn"
# pstn.plot_dot_graph()

# # #Find strongly controllable schedule
# result = paris(pstn)

#plan.temporal_network.save_as_json("test")

# print("Plan is temporally consistent:", plan.temporal_network.floyd_warshall())
# plan.temporal_network.make_minimal()
# plan.temporal_network.print_dot_graph()