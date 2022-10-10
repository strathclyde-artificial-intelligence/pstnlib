from otpl.pddl.parser import Parser
from otpl.plans.temporal_plan import PlanTemporalNetwork
from pstnlib.temporal_networks.temporal_network import TemporalNetwork
from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
from pstnlib.temporal_networks.correlation import Correlation
from pstnlib.temporal_networks.constraint import Constraint
import numpy as np
from time import time
import json
import random
import os

domain = "temporal-planning-domains/individual_experiment/domain.pddl"
problem = "temporal-planning-domains/individual_experiment/instances/instance-1.pddl"
plan_f = "temporal-planning-domains/individual_experiment/plans/rovers_instance-1_plan.pddl"
output_dir = "temporal-planning-domains/individual_experiment/networks/"
uncertainties = "temporal-planning-domains/individual_experiment/uncertainties/"
plans = ["rovers_instance-1_plan.pddl"]
factor = 1.0

instance = plan_f.split("/")[-1]
instance = instance.split(".")[0]
pddl_parser = Parser()
pddl_parser.parse_file(domain)
pddl_parser.parse_file(problem)

problem, domain = pddl_parser.problem, pddl_parser.domain
# Extracts list of and action names
actions = [a for a in domain.operators]

# Generates uncertainties. Uncertainties are added as a fraction of the duration  given in the plan.
# If action takes 10 in plan and mean_fraction is 0.8, then the mean is 8.
mean_uncertainty_factor  = np.linspace(0.6, 1.4, 5)
sd_uncertainty_factor = np.linspace(0.05, 0.25, 5)
for factor_1 in mean_uncertainty_factor:
    for factor_2 in sd_uncertainty_factor:
        uncertainties_f = {"actions": [], "tils": []}
        for action in actions:
            to_add = {"name": action, "mean_fraction": factor_1, "sd_fraction": factor_2}
            uncertainties_f["actions"].append(to_add)
            with open(uncertainties + "uncertainties_mean_{}_sd_{}.json".format(("").join(str(round(factor_1, 1)).split(".")), ("").join(str(round(factor_2, 2)).split("."))), "w") as f:
                data = json.dump(uncertainties_f, f, indent=4, separators=(", ", ": "))

# parses plan and outputs simple temporal network.
plan = PlanTemporalNetwork(pddl_parser.domain, pddl_parser.problem)
plan.read_from_file(plan_f)
deadline = plan.time_sorted_happenings[-1].time * 3
consistent = True
stns = []
# Makes stns with carrying deadlines to vary solvability/robustness
while consistent == True:
    # parses simple temporal network and makes instance of temporal network
    network = TemporalNetwork()
    network.parse_from_temporal_plan_network(plan.temporal_network)

    # Adds a deadline to stop end time-points from taking inf value.
    start_tp = network.get_timepoint_by_id(0)
    for timepoint in network.time_points:
        if not network.get_outgoing_edge_from_timepoint(timepoint):
            network.add_constraint(Constraint(start_tp, timepoint, "Overall deadline", {"lb": deadline * (1 - factor) , "ub": deadline * factor}))
    if network.check_consistency() == False:
        consistent = False
    else:
        # Computes APSP.
        network.floyd_warshall()
        network.name = instance + "_deadline_{}".format(("").join(str(round(factor, 1)).split(".")))
        factor -= 0.1
        stns.append(network)

# Samples random correlated constraints
stn = TemporalNetwork()
stn.parse_from_temporal_plan_network(plan.temporal_network)
cstn = CorrelatedTemporalNetwork()
cstn.parse_from_temporal_network(stn)
cstn.parse_uncertainties_from_json("temporal-planning-domains/individual_experiment/uncertainties/uncertainties_mean_14_sd_025")
sample = random.sample(cstn.get_probabilistic_constraints(), 5)
constraint_names = [c.get_description() for c in sample]
corr = Correlation(sample)
corr.add_random_correlation()
correlation = corr.correlation

# Makes cstns
for stn in stns:
    for uncertainty in os.listdir(uncertainties):
        cstn = CorrelatedTemporalNetwork()
        cstn.parse_from_temporal_network(stn)
        cstn.parse_uncertainties_from_json(uncertainties + uncertainty)
        constraints = [c for c in cstn.constraints if c.get_description() in constraint_names]
        corr = Correlation(constraints)
        corr.add_correlation(correlation)
        cstn.add_correlation(corr)
        cstn.name = stn.name + "_" + uncertainty
        cstn.save_as_json(output_dir + cstn.name)



