import json
import re
import sys
from time import time
from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
import csv
import pandas as pd
import os
from otpl.pddl.parser import Parser

if __name__ == "__main__":
    """
    This script takes a solution json and prints appends results to csv file.
    """
    # command line arguments
    if len(sys.argv) != 5:
        raise ValueError("Script should take four arguments:\n\t 1. The path to the network.\n\t 2. The path to the directory containing the problems.\n\t 3. The path to the domain.\n\t 4. The path to the csv file.")
    results = []
    file_path = sys.argv[1]
    tokens = file_path.split("/")
    file = tokens[-1]
    directory = "/".join(tokens[:-2])

    # References the relevant files.
    problem_path = sys.argv[2]
    problem = problem_path + file.split("_")[1] + ".pddl"
    domain = sys.argv[3]
    csv_path = sys.argv[4]

    # Parses the PDDL files to get attributes.
    pddl_parser = Parser()
    pddl_parser.parse_file(domain)
    pddl_parser.parse_file(problem)
    n_medicines = len([k for k in pddl_parser.problem.objects_type_map if pddl_parser.problem.objects_type_map[k] == "medicine"])
    n_drones = len([k for k in pddl_parser.problem.objects_type_map if pddl_parser.problem.objects_type_map[k] == "drone"])

    # Tries to opens the results jsons, catches exception and skips if it doesn't exist.
    rmp_name = "RMP_" + file
    lp_name = "PARIS_" + file

    with open(directory + "/results/" + rmp_name) as f:
        result_rmp = json.load(f)

    with open(directory + "/results_independent/" + rmp_name) as f:
        result_ind = json.load(f)

    with open(directory + "/results/" + lp_name) as f:
        result_lp = json.load(f)

    schedules = []
    # opens the network json and loads the cstn.
    network = CorrelatedTemporalNetwork()
    network.parse_from_json(directory + "/networks/" + file)
    
    schedule_rmp = result_rmp["schedule"]
    schedule_ind = result_ind["schedule"]
    schedule_lp = result_lp["schedule"]
    
    # Gets experiment parameters from name
    result = {"Name": network.name}
    tokens = network.name.split("_")
    result["Domain"] = tokens[0]
    result["Instance"] = tokens[1][9:]
    result["No Drones"] = n_drones
    result["No Deliveries"] = n_medicines
    result["Network"] = tokens[3]
    result["Deadline"] = tokens[5]
    if network.correlations:
        result["Correlation Size"] = max([len(correlation.constraints) for correlation in network.correlations])
    else:
        result["Correlation Size"] = 0
    result["Trace"] = network.calculate_trace()
    result["Generalized Variance"] = network.calculate_generalized_variance()
    result["Largest Correlation Coefficient"] = network.get_largest_correlation_coefficient()
    result["Number of Constraints"] = len(network.constraints)

    # # Simulates schedule of both RMP with/without correlation and LP using MC and saves results to list.
    if schedule_lp == None and schedule_rmp == None and schedule_ind != None:
        schedules = schedule_ind
        result["MC Probability Booles"] = float("nan")
        result["MC Probability Independent"] = network.monte_carlo(schedules)
        result["MC Probability Correlated"] = float("nan")
        result["Theoretical Probability Booles"] = float("nan")
        result["Theoretical Probability Independent"] = result_ind["probability"]
        result["Theoretical Probability Correlated"] = float("nan")
        result["Booles Runtime"] = float("nan")
        result["Indpendent Runtime"] = result_ind["runtime"]
        result["Correlated Runtime"] =float("nan")
    elif schedule_lp != None and schedule_rmp == None and schedule_ind == None:
        schedules = schedule_lp
        result["MC Probability Booles"] = network.monte_carlo(schedules)
        result["MC Probability Independent"] = float("nan")
        result["MC Probability Correlated"] = float("nan")
        result["Theoretical Probability Booles"] = result_lp["probability"]
        result["Theoretical Probability Independent"] = float("nan")
        result["Theoretical Probability Correlated"] = float("nan")
        result["Booles Runtime"] = result_lp["runtime"]
        result["Indpendent Runtime"] = float("nan")
        result["Correlated Runtime"] = float("nan")
    elif schedule_lp == None and schedule_rmp != None and schedule_ind == None:
        schedules = schedule_rmp
        result["MC Probability Booles"] = float("nan")
        result["MC Probability Independent"] = float("nan")
        result["MC Probability Correlated"] = network.monte_carlo(schedules)
        result["Theoretical Probability Booles"] = float("nan")
        result["Theoretical Probability Independent"] = float("nan")
        result["Theoretical Probability Correlated"] = result_rmp["probability"]
        result["Booles Runtime"] = float("nan")
        result["Indpendent Runtime"] = float("nan")
        result["Correlated Runtime"] = result_rmp["runtime"]
    elif schedule_lp != None and schedule_rmp != None and schedule_ind == None:
        schedules = [schedule_lp, schedule_rmp]
        probs = network.monte_carlo(schedules)
        result["MC Probability Booles"] = probs[0]
        result["MC Probability Independent"] = float("nan")
        result["MC Probability Correlated"] = probs[1]
        result["Theoretical Probability Booles"] = result_lp["probability"]
        result["Theoretical Probability Independent"] = float("nan")
        result["Theoretical Probability Correlated"] = result_rmp["probability"]
        result["Booles Runtime"] = result_lp["runtime"]
        result["Indpendent Runtime"] = float("nan")
        result["Correlated Runtime"] = result_rmp["runtime"]
    elif schedule_lp != None and schedule_rmp == None and schedule_ind != None:
        schedules = [schedule_lp, schedule_ind]
        probs = network.monte_carlo(schedules)
        result["MC Probability Booles"] = probs[0]
        result["MC Probability Independent"] = probs[1]
        result["MC Probability Correlated"] = float("nan")
        result["Theoretical Probability Booles"] = result_lp["probability"]
        result["Theoretical Probability Independent"] = result_ind["probability"]
        result["Theoretical Probability Correlated"] = float("nan")
        result["Booles Runtime"] = result_lp["runtime"]
        result["Indpendent Runtime"] = result_ind["runtime"]
        result["Correlated Runtime"] = float("nan")
    elif schedule_lp == None and schedule_rmp != None and schedule_ind != None:
        schedules = [schedule_ind, schedule_rmp]
        probs = network.monte_carlo(schedules)
        result["MC Probability Booles"] = float("nan")
        result["MC Probability Independent"] = probs[0]
        result["MC Probability Correlated"] = probs[1]
        result["Theoretical Probability Booles"] = float("nan")
        result["Theoretical Probability Independent"] = result_ind["probability"]
        result["Theoretical Probability Correlated"] = result_rmp["probability"]
        result["Booles Runtime"] = float("nan")
        result["Indpendent Runtime"] = result_ind["runtime"]
        result["Correlated Runtime"] = result_rmp["runtime"]
    else:
        schedules = [schedule_lp, schedule_ind, schedule_rmp]
        probs = network.monte_carlo(schedules)
        result["MC Probability Booles"] = probs[0]
        result["MC Probability Independent"] = probs[1]
        result["MC Probability Correlated"] = probs[2]
        result["Theoretical Probability Booles"] = result_lp["probability"]
        result["Theoretical Probability Independent"] = result_ind["probability"]
        result["Theoretical Probability Correlated"] = result_rmp["probability"]
        result["Booles Runtime"] = result_lp["runtime"]
        result["Indpendent Runtime"] = result_ind["runtime"]
        result["Correlated Runtime"] = result_rmp["runtime"]
    results.append(result)
    
    keys = results[0].keys()
    if os.path.exists(csv_path):
        # Tries to open with write - this should raise value error if file doesnt exist.
        with open(csv_path, "a", newline='') as f:
            writer = csv.DictWriter(f, keys)
            writer.writerows(results)
    else:
        vals = result.values()
        with open(csv_path, "w+", newline='') as f:
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            writer.writerows(results)

