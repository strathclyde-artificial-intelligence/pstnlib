from distutils.errors import DistutilsPlatformError
from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
from pstnlib.optimisation.pstn_optimisation_class import PstnOptimisation
import numpy as np
import os
path = "temporal-planning-domains/rovers-metric-time-2006/"
to_solve = os.listdir(path + "networks/")

for i in range(len(to_solve)):
    #if "rovers_instance-1_deadline-68_uncertainties-4_ncorrelations-2_sizecorrelation-2" in to_solve[i]:
    if "instance-1_" in to_solve[i] and "uncertainties-7" in to_solve[i] and "ncorrelations-1" in to_solve[i] and "sizecorrelation-3" in to_solve[i]:
        file = path + "networks/" + to_solve[i]
        cstn = CorrelatedTemporalNetwork()
        try:
            cstn.parse_from_json(file)
            op = PstnOptimisation(cstn)
            op.optimise()
            # Saves initial LP solution if it exists.
            for solution in op.solutions:
                if "PARIS" in solution.model.getAttr("ModelName"):
                    lp = solution
                    solution.to_json(path + "results/")
            # Saves final RMP solution.
            rmp = op.solutions[-1]
            op.solutions[-1].to_json(path + "results/")
            # gets schedules and simulates execution
            schedules = [r.get_schedule() for r in [lp, rmp]]
            experimental_probs = cstn.monte_carlo(schedules)
            #print(experimental_probs)
            #print(cstn.monte_carlo(schedules[0]))
            #print(cstn.monte_carlo(schedules[1]))
            print("\n", to_solve[i])
            print("LP: ")
            print("\tTheoretical Probability: ", lp.get_probability())
            print("\tExperimental Probability: ", cstn.monte_carlo(schedules[0]))
            print("RMP: ")
            print("\tTheoretical Probability: ", rmp.get_probability())
            print("\tExperimental Probability: ", cstn.monte_carlo(schedules[1]))
            print("\n")
        except:
            continue