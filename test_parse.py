from otpl.pddl.parser import Parser
from otpl.pddl.grounding import Grounding
from otpl.plans.temporal_plan import PlanTemporalNetwork
from otpl.temporal_networks.simple_temporal_network import SimpleTemporalNetwork

domain = "temporal-planning-domains/map-analyzer-temporal-satisficing-2014/domain.pddl"
problem = "temporal-planning-domains/map-analyzer-temporal-satisficing-2014/instances/instance-1.pddl"
planfile = "temporal-planning-domains/map-analyzer-temporal-satisficing-2014/plans/map_analyzer_instance-1_plan.pddl.1"

# parse PDDL domain and problem files
print("Parsing PDDL domain and problem file...")
pddl_parser = Parser()
pddl_parser.parse_file(domain)
pddl_parser.parse_file(problem)
    
print("Parsing PDDL plan file...")
plan = PlanTemporalNetwork(pddl_parser.domain, pddl_parser.problem)
plan.read_from_file(planfile)
plan.temporal_network.print_graph_as_json()