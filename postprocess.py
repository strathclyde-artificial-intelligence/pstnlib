import os
from matplotlib import pyplot as plt
import json
from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
import numpy as np
import csv

def postprocess(directory: str):
    """
    Parses results files in a directory and plot probability for column generation versus LP.
    """
    results = []
    results_dir = directory + "results/"
    networks_dir = directory + "networks/"

    for file in os.listdir(networks_dir):
        #print("Parsing result {} of {}.".format(os.listdir(networks_dir).index(file), len(os.listdir(networks_dir))))
        if file == "rovers_instance-1_plan_2_uncertainties_2_correlationsize4.json":
            try:
                # Tries to opens the results jsons, catches exception and skips if it doesn't exist.
                
                rmp_name = "RMP_" + file
                lp_name = "PARIS_" + file

                with open(results_dir + rmp_name) as f:
                    result_rmp = json.load(f)

                with open(results_dir + lp_name) as f:
                    result_lp = json.load(f)

                schedules = []
                # opens the network json and loads the cstn.
                network = CorrelatedTemporalNetwork()
                network.parse_from_json(networks_dir + file)
                network.plot_dot_graph()
                
    #             schedule_rmp = result_rmp["schedule"]
    #             schedule_lp = result_lp["schedule"]
    #             print("\nRMP: ", schedule_rmp)
    #             print("\nLP: ", schedule_lp)
    #             diff = {}
    #             for key in schedule_rmp:
    #                 diff[key] = schedule_rmp[key] - schedule_lp[key]
    #             print("\nDiff: ", diff)
                
    #             # Gets experiment parameters from name
    #             result = {"Name": network.name}
    #             tokens = network.name.split("_")
    #             result["Domain"] = tokens[0]
    #             result["Instance"] = tokens[1][-1]
    #             result["Network"] = tokens[3]
    #             result["Uncertainties"] = tokens[5]
    #             result["Correlation Size"] = tokens[6][-1]
    #             result["Eta"] = [c.eta_used for c in network.correlations]

    #             # # Simulates schedule of both RMP and LP using MC and saves results to list.
    #             if schedule_lp == None and schedule_rmp != None:
    #                 schedules = schedule_rmp
    #                 result["MC Probability LP"] = 0
    #                 result["MC Probability RMP"] = network.monte_carlo(schedules)
    #                 result["Theoretical Probability LP"] = 0
    #                 result["Theoretical Probability RMP"] = result_rmp["probability"]
    #                 result["LP Runtime"] = 0
    #                 result["RMP Runtime"] = result_rmp["runtime"]

    #             elif schedule_lp != None and schedule_rmp == None:
    #                 schedules = schedule_lp
    #                 result["MC Probability LP"] = network.monte_carlo(schedules)
    #                 result["MC Probability RMP"] = 0
    #                 result["Theoretical Probability LP"] = result_lp["probability"]
    #                 result["Theoretical Probability RMP"] = 0
    #                 result["LP Runtime"] = result_lp["runtime"]
    #                 result["RMP Runtime"] = 0
    #             else:
    #                 schedules = [schedule_rmp, schedule_lp]
    #                 probs = network.monte_carlo(schedules)
    #                 result["MC Probability LP"] = probs[1]
    #                 result["MC Probability RMP"] = probs[0]
    #                 result["Theoretical Probability LP"] = result_lp["probability"]
    #                 result["Theoretical Probability RMP"] = result_rmp["probability"]
    #                 result["LP Runtime"] = result_lp["runtime"]
    #                 result["RMP Runtime"] = result_rmp["runtime"]

    #             result["MC probability Delta"] = result["MC Probability RMP"] - result["MC Probability LP"]
    #             result["Theoretical Probability Delta"] = result["Theoretical Probability RMP"] - result["Theoretical Probability LP"]
    #             result["Runtime Delta"] = result["RMP Runtime"] - result["LP Runtime"]

    #             results.append(result)
    #             print("\n")
    #             print(results)
            except FileNotFoundError:
                continue
    # print("Complete, saving to csv")

    # keys = results[0].keys()
    # with open("junk/results.csv", "w", newline='') as f:
    #     writer = csv.DictWriter(f, keys)
    #     writer.writeheader()
    #     writer.writerows(results)

    # # Makes plots comparing Monte-Carlo probability and theoretical probability
    # plt.figure()
    # plt.scatter(probability_lp, vals_lp, label="LP", marker ="x", color="blue")
    # plt.scatter(probability_rmp, vals_rmp, label ="RMP", marker =".", color="orange")
    # plt.legend(fontsize=9)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.xlabel("MC Probability")
    # plt.ylabel("Theoretical Probability")
    # plt.savefig("junk/probs_test.png")


if __name__ == "__main__":
    directory = "temporal-planning-domains/rovers-metric-time-2006/"
    postprocess(directory)