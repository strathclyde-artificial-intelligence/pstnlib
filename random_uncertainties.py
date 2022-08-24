import random
import numpy as np
from scipy import stats
from otpl.pddl.parser import Parser
import json

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
            file = file + ".json"
    uncertainties = generate_random_uncertainties(domain_file, problem_file)
    with open(output_file, "w") as f:
        data = json.dump(uncertainties, f, indent=4, separators=(", ", ": "))
