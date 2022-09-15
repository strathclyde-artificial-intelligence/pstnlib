from random_generation import save_random_uncertainties, sample_probabilistic_constraints
import os
from otpl.plans.temporal_plan import PlanTemporalNetwork
from otpl.pddl.parser import Parser
from pstnlib.temporal_networks.temporal_network import TemporalNetwork
from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
from pstnlib.temporal_networks.constraint import Constraint
from pstnlib.temporal_networks.correlation import Correlation
import random

directory = "temporal-planning-domains/"
# # Gets list of directories containing domains.
# subdirectories = [d for d in os.listdir(directory) if d not in ["README.md", ".git", 'pathways-metric-time-2006', "tpp-metric-time-2006"]]

# successful = []
# for d in subdirectories:
#     print("\nTesting domain ", d)
#     try:
#         path = directory + d
#         domain = path + "/domain.pddl"
#         problem = path + "/instances/instance-1.pddl"
#         # Generates random uncertainties for each action given domain and problem.
#         for i in range(1, 10):
#             save_random_uncertainties(domain, problem, path + "/uncertainties/uncertainties_{}".format(i))
#         successful.append(d)
#     except:
#         continue

#successful = ['turn-and-open-temporal-satisficing-2014', 'parking-temporal-satisficing-2014', 'crew-planning-temporal-satisficing-2011', 'road-traffic-accident-management-temporal-satisficing-2014', 'rovers-metric-time-2006', 'peg-solitaire-temporal-satisficing-2011', 'map-analyzer-temporal-satisficing-2014', 'storage-temporal-satisficing-2014', 'driver-log-temporal-satisficing-2014', 'satellite-temporal-satisficing-2014', 'sokoban-temporal-satisficing-2011', 'temporal-machine-shop-temporal-satisficing-2014', 'floor-tile-temporal-satisficing-2014', 'elevator-temporal-satisficing-2011', 'match-cellar-temporal-satisficing-2014']
successful = ['rovers-metric-time-2006']
for d in successful:
    pddl_parser = Parser()
    path = directory + d
    domain = path + "/domain.pddl"
    plans = path + "/plans/"
    # Loops through all plans in the directory
    for file in os.listdir(directory + d + "/plans"):
        for token in file.split("_"):
            if "instance" in token:
                problem_id = token
                problem = path + "/instances/" + "{}.pddl".format(token)
        tokens = file.split(".")
        # If there is more than one plan for this instance it changes the problem id acdordingly so as not to overwrite existing file.
        if len(tokens) > 2:
            problem_id = problem_id + "-" + tokens[-1]
        pddl_parser.parse_file(domain)
        pddl_parser.parse_file(problem)
        # parses plan and makes instance of temporal plan
        plan = PlanTemporalNetwork(pddl_parser.domain, pddl_parser.problem)
        # Gets overall plan deadline from time sorted happenings
        plan.read_from_file(plans + file)
        # makes a number of deadlines to modify solvability/probability
        deadlines = [f * plan.time_sorted_happenings[-1].time for f in [0.6, 0.8, 1.0, 1.2, 1.4]]
        for deadline in deadlines:
            tokens = str(deadline).split(".")
            deadline_id = tokens[0]
            # loops through uncertainty jsons.
            for uncertainties in os.listdir(path + "/uncertainties"):
                uncertainties_id = uncertainties.split("_")[1][:-5]
                uncertainties_file = path + "/uncertainties/" + uncertainties
                partitions = [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)]
                # Samples random constraints to make correlated.
                for partition in partitions:
                    try:
                        # creates temporal network and parses from simple temporal network.
                        network = TemporalNetwork()
                        network.parse_from_temporal_plan_network(plan.temporal_network)
                        # Gets start and end time-points and constrains by deadline.
                        start_tp = network.get_timepoint_by_id(0)
                        for timepoint in network.time_points:
                            if not network.get_outgoing_edge_from_timepoint(timepoint):
                                network.add_constraint(Constraint(start_tp, timepoint, "Overall deadline", {"lb": 0, "ub": deadline}))
                        # Parses temporal network to make a correlated temporal network.
                        cstn = CorrelatedTemporalNetwork()
                        cstn.parse_from_temporal_network(network)
                        cstn.parse_uncertainties_from_json(uncertainties_file)
                        # Samples a number of probabilistic constraints according to partition.
                        correlated_constraints = sample_probabilistic_constraints(cstn, partition[0], partition[1])
                        for item in correlated_constraints:
                            corr = Correlation(item)
                            # Adds a random correlation matrix and adds correlation to the network.
                            corr.add_random_correlation(eta = random.uniform(0, 1))
                            cstn.add_correlation(corr)
                        cstn.name = "rovers_{}_deadline-{}_uncertainties-{}_ncorrelations-{}_sizecorrelation-{}".format(problem_id, deadline_id, uncertainties_id, partition[0], partition[1])
                        cstn.save_as_json(path + "/networks/" + cstn.name)
                    except ValueError:
                        continue
                    

            



