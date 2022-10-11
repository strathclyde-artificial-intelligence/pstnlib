import json
import re
import sys
from time import time
from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
import csv
import pandas as pd
import os

if __name__ == "__main__":
    """
    This script takes a solution json and prints appends results to csv file.
    """
    # command line arguments
    if len(sys.argv) != 3:
        raise ValueError("Script should take two arguments:\n\t 1. The path to the network.\n\t 2. The path to the csv file.")
    results = []
    file_path = sys.argv[1]
    tokens = file_path.split("/")
    file = tokens[-1]
    directory = "/".join(tokens[:-2])

    csv_path = sys.argv[2]

    # Tries to opens the results jsons, catches exception and skips if it doesn't exist.
    rmp_name = "RMP_" + file
    lp_name = "PARIS_" + file

    with open(directory + "/results/" + rmp_name) as f:
        result_rmp = json.load(f)

    with open(directory + "/results/" + lp_name) as f:
        result_lp = json.load(f)

    schedules = []
    # opens the network json and loads the cstn.
    network = CorrelatedTemporalNetwork()
    network.parse_from_json(directory + "/networks/" + file)
    
    schedule_rmp = result_rmp["schedule"]
    schedule_lp = result_lp["schedule"]
    
    # Gets experiment parameters from name
    result = {"Name": network.name}
    tokens = network.name.split("_")
    result["Domain"] = tokens[0]
    result["Instance"] = tokens[1][-1]
    result["Network"] = tokens[3]
    result["Deadline"] = tokens[5]
    result["Trace"] = network.calculate_trace()
    result["Generalized Variance"] = network.calculate_generalized_variance()

    # Simulates schedule of both RMP and LP using MC and saves results to list.
    if schedule_lp == None and schedule_rmp != None:
        schedules = schedule_rmp
        result["MC Probability LP"] = 0
        result["MC Probability RMP"] = network.monte_carlo(schedules)
        result["Theoretical Probability LP"] = 0
        result["Theoretical Probability RMP"] = result_rmp["probability"]
        result["LP Runtime"] = 0
        result["RMP Runtime"] = result_rmp["runtime"]

    elif schedule_lp != None and schedule_rmp == None:
        schedules = schedule_lp
        result["MC Probability LP"] = network.monte_carlo(schedules)
        result["MC Probability RMP"] = 0
        result["Theoretical Probability LP"] = result_lp["probability"]
        result["Theoretical Probability RMP"] = 0
        result["LP Runtime"] = result_lp["runtime"]
        result["RMP Runtime"] = 0
    else:
        schedules = [schedule_rmp, schedule_lp]
        probs = network.monte_carlo(schedules)
        result["MC Probability LP"] = probs[1]
        result["MC Probability RMP"] = probs[0]
        result["Theoretical Probability LP"] = result_lp["probability"]
        result["Theoretical Probability RMP"] = result_rmp["probability"]
        result["LP Runtime"] = result_lp["runtime"]
        result["RMP Runtime"] = result_rmp["runtime"]

    result["MC probability Delta"] = result["MC Probability RMP"] - result["MC Probability LP"]
    result["Theoretical Probability Delta"] = result["Theoretical Probability RMP"] - result["Theoretical Probability LP"]
    result["Runtime Delta"] = result["RMP Runtime"] - result["LP Runtime"]
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

