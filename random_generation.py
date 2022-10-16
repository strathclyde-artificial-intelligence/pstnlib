import random
from xml.dom.minidom import Attr
import numpy as np
from scipy import stats
from otpl.pddl.parser import Parser
import json
from pstnlib.temporal_networks.constraint import Constraint
from pstnlib.temporal_networks.timepoint import TimePoint
from pstnlib.temporal_networks.correlation import Correlation
from pstnlib.temporal_networks.temporal_network import TemporalNetwork
from pstnlib.temporal_networks.probabilistic_temporal_network import ProbabilisticTemporalNetwork
from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
from otpl.plans.temporal_plan import PlanTemporalNetwork
inf = 1000000000

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

def generate_random_constraints(network: TemporalNetwork, deadline: float, number: int) -> list[Constraint]:
    """
    Randomly generates a set of constraints of size n to add to the temporal network. 
    """
    cs = [c for c in network.constraints if "Ordering" in c.label or "Interference" in c.label]
    sample = random.sample(cs, number)
    consistent = False
    while consistent == False:
        for constraint in sample:
            ub = random.uniform(0, deadline)
            lb = random.uniform(0, ub)
            constraint.duration_bound = {"lb": lb, "ub": ub}
        if network.check_consistency() == True:
            consistent = True
    return network

def generate_random_cstn(domain_f: str, problem_f: str, plan_f: str):
    """
    Generates a random simple temporal network in all pairs shortest path form from a given pddl domain, problem and plan.
    """
    instance = plan_f.split("/")[-1]
    instance = instance.split(".")[0]
    pddl_parser = Parser()
    pddl_parser.parse_file(domain_f)
    pddl_parser.parse_file(problem_f)

    # parses plan and outputs simple temporal network.
    plan = PlanTemporalNetwork(pddl_parser.domain, pddl_parser.problem)
    plan.read_from_file(plan_f)
    deadline = plan.time_sorted_happenings[-1].time

    # parses simple temporal network and makes instance of temporal network
    network = CorrelatedTemporalNetwork()
    network.parse_from_temporal_plan_network(plan.temporal_network)

    # Generates and adds random uncertainties.
    uncertainties = generate_random_uncertainties(domain_f, problem_f)
    network.parse_uncertainties_from_dict(uncertainties)

    # If both start and end timepoints are uncontrollable it decomposes the constraint containing the beginning timepoint.
    for constraint in network.get_requirement_constraints():
        incoming = network.get_incoming_probabilistic(constraint)
        if incoming["start"] != None and incoming["end"] != None:
            # Make new controllable timepoint
            index = max([t.id for t in network.time_points]) + 1
            intermediate_timepoint_start = TimePoint(index, "Intermediate timepoint {}".format(index))
            network.add_time_point(intermediate_timepoint_start)
            # Udates probabilistic sink to new timepoint
            incoming["start"].sink = intermediate_timepoint_start
            # Makes constraint between probabilistic sink and uncontrollable start
            network.add_constraint(Constraint(intermediate_timepoint_start, constraint.source, "Intermediate constraint added for {}".format(constraint.get_description()), {"lb": 0, "ub": inf}))

    for constraint in network.constraints:
        if "Deadline" in constraint.label:
            constraint.duration_bound = {"lb": 0, "ub": deadline * 1.5}

    # Generates and adds random correlation.
    correlation_size = random.randint(2, 6)
    correlated_constraints = sample_probabilistic_constraints(network, 1, correlation_size)[0]
    corr = Correlation(correlated_constraints)
    corr.add_random_correlation(eta = random.uniform(0.0, 1.0))
    network.add_correlation(corr)
    return network


