import random
import numpy as np
from scipy import stats
from otpl.pddl.parser import Parser
import json
from pstnlib.temporal_networks.constraint import Constraint
from pstnlib.temporal_networks.temporal_network import TemporalNetwork
from pstnlib.temporal_networks.probabilistic_temporal_network import ProbabilisticTemporalNetwork
from otpl.pddl.parser import Parser
from otpl.plans.temporal_plan import PlanTemporalNetwork

def generate_random_uncertainties(domain_file: str, problem_file: str) -> dict:
    """
    randomly generates a dictionary of uncertainties for action and til durations.
    """
    # Parses domain and problem file
    pddl_parser = Parser()
    pddl_parser.parse_file(domain_file)
    pddl_parser.parse_file(problem_file)
    problem, domain = pddl_parser.problem, pddl_parser.domain
    
    # Extracts list of tils and action names
    tils = problem.timed_initial_literals
    actions = [a for a in domain.operators]

    # Randomly selects tils to be made probabilistic
    #no_to_be_randomised = random.randint(min(1, len(tils)), len(tils))
    no_to_be_randomised = len(tils)
    random_tils = random.sample(tils, no_to_be_randomised)

    # Randomly selects actions to be made probabilistic
    #no_to_be_randomised = random.randint(min(1, len(actions)), len(actions))
    no_to_be_randomised = len(actions)
    random_actions = random.sample(actions, no_to_be_randomised)

    # Randomly generates uncertainties. Uncertainties are added as a fraction of the duration  given in the plan.
    # If action takes 10 in plan and mean_fraction is 0.8, then the mean is 8.
    uncertainties = {"actions": [], "tils": []}
    for action in random_actions:
        to_add = {"name": action, "mean_fraction": abs(np.random.normal(1, 0.2)), "sd_fraction": abs(np.random.normal(0.15, 0.05))}
        uncertainties["actions"].append(to_add)
    
    for til in random_tils:
        to_add = {"name": til, "mean_fraction": abs(np.random.normal(1, 0.2)), "sd_fraction": abs(np.random.normal(0.15, 0.05))}
    return uncertainties

def save_random_uncertainties(domain_file: str, problem_file: str, output_file: str) -> None:
    """
    wrapper for generate_random_uncertainties to save as json
    """
    if output_file[-5:] != ".json":
            output_file = output_file + ".json"
    uncertainties = generate_random_uncertainties(domain_file, problem_file)
    with open(output_file, "w") as f:
        data = json.dump(uncertainties, f, indent=4, separators=(", ", ": "))

def sample_probabilistic_constraints(network: ProbabilisticTemporalNetwork, n_correlations: int, size: int) -> list[Constraint]:
    """
    Randomly generates "n_correlations" samples of the probabilistic constraints in the network. Size is the number of constraints in each sample.
    """
    if len(network.get_probabilistic_constraints()) < n_correlations * size:
        raise ValueError("Insuffucient probabilistic constraints for required size.")
    sample = random.sample(network.get_probabilistic_constraints(), n_correlations * size)
    random.shuffle(sample)
    partitions = [sample[i::n_correlations] for i in range(n_correlations)]
    return partitions

def generate_random_constraints(network: TemporalNetwork, deadline: float, size: int) -> list[Constraint]:
    """
    Randomly generates a set of constraints of size n to add to the temporal network. 
    """
    # Makes list of constraints not related to actions.
    cs = []
    for c in network.constraints:
        if "Ordering" in c.label or "Interference" in c.label:
            cs.append(c)

    sample = random.sample(cs, size)
    consistent = False
    while consistent == False:
        for constraint in sample:
            ub = random.uniform(0, deadline)
            lb = random.uniform(0, ub)
            constraint.duration_bound = {"lb": lb, "ub": ub}
        if network.floyd_warshall()[1] == True:
            consistent = True
    return network

def generate_random_stns(domain_f: str, problem_f: str, plan_f: str, number: int, output_dir: str):
    """
    Generates number of randomly generated simple temporal networks given a plan file.
    """
    instance = plan_f.split("/")[-1]
    instance = instance.split(".")[0]
    networks = []
    for i in range(number):
        pddl_parser = Parser()
        pddl_parser.parse_file(domain_f)
        pddl_parser.parse_file(problem_f)

        # parses plan and outputs simple temporal network.
        plan = PlanTemporalNetwork(pddl_parser.domain, pddl_parser.problem)
        plan.read_from_file(plan_f)
        deadline = plan.time_sorted_happenings[-1].time

        # parses simple temporal network and makes instance of temporal network
        network = TemporalNetwork()
        network.parse_from_temporal_plan_network(plan.temporal_network)

        # Adds a deadline to stop end time-points from taking inf value.
        start_tp = network.get_timepoint_by_id(0)
        for timepoint in network.time_points:
            if not network.get_outgoing_edge_from_timepoint(timepoint):
                network.add_constraint(Constraint(start_tp, timepoint, "Overall deadline", {"lb": 0, "ub": deadline * 1.5}))

        n = len([c for c in network.constraints if "Ordering" in c.label or "Interference" in c.label])
        network = generate_random_constraints(network, deadline, random.randint(2, 10))
        network.name = instance + "_{}".format(i + 1)
        networks.append(network)
    return networks
    

