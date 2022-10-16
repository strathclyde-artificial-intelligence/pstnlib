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
from random_generation import generate_random_cstn
import numpy as np
from time import time
from graphviz import Digraph
import os
import random

domain = "temporal-planning-domains/satellite-temporal-satisficing-2014/domain.pddl"
problem_dir = "temporal-planning-domains/satellite-temporal-satisficing-2014/instances/"
plan_dir = "temporal-planning-domains/satellite-temporal-satisficing-2014/plans/"
output_dir = "temporal-planning-domains/satellite-temporal-satisficing-2014/networks/"
no_cstns = 20
problems = ["instance-1.pddl", "instance-2.pddl", "instance-3.pddl", "instance-4.pddl", "instance-5.pddl"]
deadline_factors = np.array([0.6, 0.7, 0.8, 0.9, 1.0])

for problem in problems:
    tokens = problem.split(".")
    plan = "satellite_{}_plan.pddl".format(tokens[0])
    for i in range(1, no_cstns + 1):
        network = generate_random_cstn(domain, problem_dir + problem, plan_dir + plan)
        for factor in deadline_factors:
            cstn = network.copy()
            cstn.name = "satellite_{}_network_{}_deadline_{}".format(problem.split(".")[0], i, ("").join(str(round(factor, 1)).split(".")))
            for constraint in cstn.constraints:
                if "Deadline" in constraint.label:
                    bound = constraint.ub
                    constraint.duration_bound = {'lb': (1 - factor) * bound, 'ub': factor * bound}
            cstn.save_as_json(output_dir + cstn.name + ".json")
            
                
