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
    no_to_be_randomised = random.randint(min(1, len(tils)), len(tils))
    random_tils = random.sample(tils, no_to_be_randomised)

    # Randomly selects actions to be made probabilistic
    no_to_be_randomised = random.randint(min(1, len(actions)), len(actions))
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

def generate_random_correlation(n, eta, size=1):
    """
    Description:    Code for generating random positive semidefinite correlation matrices. Taken from https://gist.github.com/junpenglao/b2467bb3ad08ea9936ddd58079412c1a
                    based on code from "Generating random correlation matrices based on vines and extended onion method", Daniel Lewandowski, Dorots Kurowicka and Harry Joe, 2009.
    Input:          n:      Size of correlation matrix
                    eta:    Parameter - the larger eta is, the closer to the identity matrix will be the correlation matrix (more details see https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices)
                    size:   Number of samples
    Output:         Correlation matrix
    """
    beta0 = eta - 1 + n/2
    shape = n * (n-1) // 2
    triu_ind = np.triu_indices(n, 1)
    beta_ = np.array([beta0 - k/2 for k in triu_ind[0]])
    # partial correlations sampled from beta dist.
    P = np.ones((n, n) + (size,))
    P[triu_ind] = stats.beta.rvs(a=beta_, b=beta_, size=(size,) + (shape,)).T
    # scale partial correlation matrix to [-1, 1]
    P = (P-.5)*2
    
    for k, i in zip(triu_ind[0], triu_ind[1]):
        p = P[k, i]
        for l in range(k-1, -1, -1):  # convert partial correlation to raw correlation
            p = p * np.sqrt((1 - P[l, i]**2) *
                            (1 - P[l, k]**2)) + P[l, i] * P[l, k]
        P[k, i] = p
        P[i, k] = p
    return np.transpose(P, (2, 0 ,1))[0]
