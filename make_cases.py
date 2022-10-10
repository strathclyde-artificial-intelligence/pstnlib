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
from random_generation import generate_random_constraints, generate_random_stn, sample_probabilistic_constraints, save_random_uncertainties
import numpy as np
from time import time
from graphviz import Digraph
import os
import random

domain = "temporal-planning-domains/crew-planning-temporal-satisficing-2011/domain.pddl"
problem = "temporal-planning-domains/crew-planning-temporal-satisficing-2011/instances/instance-1.pddl"
plan_dir = "temporal-planning-domains/crew-planning-temporal-satisficing-2011/plans/"
output_dir = "temporal-planning-domains/crew-planning-temporal-satisficing-2011/networks/"
uncertainties = "temporal-planning-domains/crew-planning-temporal-satisficing-2011/uncertainties/"
count = 1
plans = ["crewplanning_instance-1_plan.pddl"]

# for i in range(1, 6):
#     save_random_uncertainties(domain, problem, uncertainties + "uncertainties_{}".format(i))

for plan in plans:
    for k in range(2, 12, 2):
        stn = generate_random_stn(domain, problem, plan_dir + plan, k)
        stn.name = stn.name + "_{}".format(count)
        count += 1
        for uncert in os.listdir(uncertainties):
            token = uncert[:-5]
            for i in range(2, 7):
                # Adds a random correlation matrix and adds correlation to the network.
                for j in np.linspace(0.1, 0.9, 5):
                    cstn = CorrelatedTemporalNetwork()
                    cstn.parse_from_temporal_network(stn)
                    cstn.parse_uncertainties_from_json(uncertainties + uncert)
                    correlated_constraints = sample_probabilistic_constraints(cstn, 1, i)
                    for item in correlated_constraints:
                        corr = Correlation(item)
                        corr.add_random_correlation(eta = j)
                    cstn.add_correlation(corr)
                    cstn.name = stn.name + "_" + token + "_correlationsize_{}".format(i) + "_eta_{}".format(("").join(str(round(j, 1)).split(".")))
                    cstn.save_as_json(output_dir + cstn.name)