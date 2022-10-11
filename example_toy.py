from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
from pstnlib.temporal_networks.timepoint import TimePoint
from pstnlib.temporal_networks.constraint import Constraint, ProbabilisticConstraint
from pstnlib.optimisation.pstn_optimisation_class import PstnOptimisation
from pstnlib.temporal_networks.correlation import Correlation
import numpy as np
import csv
from time import time

inf = 1e9

sd1 = 10
sd2 = 25
ub = 200

results = []
correlation_coefficients = np.array([0.3, 0.6, 0.9])
deadline_factors = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
deadline_move = np.array([-20, -10, 0, 10, 20])

indpendent_results = []
# Solves assuming independence:
for deadline in deadline_factors:
    for item in deadline_move:
        # # Makes timepoints and constraints.
        b0 = TimePoint(0, "Begin Travel 1")
        b1 = TimePoint(1, "End Travel 1")
        b2 = TimePoint(2, "Begin Travel 2")
        b3 = TimePoint(3, "End Travel 2")
        c1 = ProbabilisticConstraint(b0, b1, "Travel 1", {"mean": 60, "sd": sd1})
        c2 = Constraint(b1, b2, "Collect", {"lb": 0, "ub": inf})
        c3 = ProbabilisticConstraint(b2, b3, "Travel 2", {"mean": 100, "sd": sd2})
        c4 = Constraint(b0, b3, "Deadline", {"lb": ub * (1 - deadline) + item, "ub": ub * deadline + item})

        # Makes toy stn from paper while varying parameters
        cstn = CorrelatedTemporalNetwork()
        cstn.time_points = [b0, b1, b2, b3]
        cstn.constraints = [c1, c2, c3, c4]
        cstn.name = "toy_corr_0_deadline_{}_sd_{}".format(("").join(str(round(deadline, 2)).split(".")), ("").join(str(round(item, 2)).split(".")))
        cstn.save_as_json("temporal-planning-domains/toy/networks/{}".format(cstn.name))
        # Solves using column generation and gets schedule
        op = PstnOptimisation(cstn, verbose=True)
        start_t = time()
        op.optimise()
        runtime = time() - start_t
        result_dic = {}
        result = op.solutions[-1]
        schedule = result.get_schedule()
        result_dic["deadline factor"] = deadline
        result_dic["deadline move"] = item
        result_dic["schedule"] = schedule
        result_dic["delta"] = schedule["2"] - schedule["0"]
        result_dic["probability"] = result.get_probability()
        result_dic["runtime"] = runtime
        # Gets monte-carlo probability
        indpendent_results.append(result_dic)

# for coeff in correlation_coefficients:
#     for deadline in deadline_factors:
#         for sd in sd_factors:
#             result = {}
#             # # Makes timepoints and constraints.
#             b0 = TimePoint(0, "Begin Travel 1")
#             b1 = TimePoint(1, "End Travel 1")
#             b2 = TimePoint(2, "Begin Travel 2")
#             b3 = TimePoint(3, "End Travel 2")
#             c1 = ProbabilisticConstraint(b0, b1, "Travel 1", {"mean": 60, "sd": sd1 * sd})
#             c2 = Constraint(b1, b2, "Collect", {"lb": 0, "ub": inf})
#             c3 = ProbabilisticConstraint(b2, b3, "Travel 2", {"mean": 100, "sd": sd2 * sd})
#             c4 = Constraint(b0, b3, "Deadline", {"lb": ub * (1 - deadline), "ub": ub * deadline})

#             # Makes toy stn from paper while varying parameters
#             cstn = CorrelatedTemporalNetwork()
#             cstn.time_points = [b0, b1, b2, b3]
#             cstn.constraints = [c1, c2, c3, c4]
#             if coeff != 0:
#                 corr = Correlation([c1, c3])
#                 corr.add_correlation(np.array([[1, coeff],[coeff, 1]]))
#                 cstn.add_correlation(corr)

#             cstn.name = "toy_corr_{}_deadline_{}_sd_{}".format(("").join(str(round(coeff, 2)).split(".")), ("").join(str(round(deadline, 2)).split(".")), ("").join(str(round(sd, 2)).split(".")))
#             cstn.save_as_json("temporal-planning-domains/toy/networks/{}".format(cstn.name))
#             result["name"] = cstn.name
#             result["correlation coefficient"] = coeff
#             result["deadline factor"] = deadline
#             result["sd"] = sd
#             for item in indpendent_results:
#                 if item["deadline factor"] == deadline and item["sd factor"] == sd:
#                     result["independent probability"] = item["probability"]
#                     result["independent delta"] = item["delta"]
#                     result["independent runtime"] = item["runtime"]
#                     schedule_to_compare = item["schedule"]

#             # Solves using column generation and gets schedule
#             op = PstnOptimisation(cstn, verbose=True)
#             start_t = time()
#             op.optimise()
#             runtime = time() - start_t
#             convex = op.solutions[-1]
#             schedule = convex.get_schedule()
#             result["correlated delta"] = schedule["2"] - schedule["0"]
#             result["correlated probability"] = convex.get_probability()
#             result["correlated runtime"] = runtime
#             # Gets monte-carlo probability
#             mc_probabilities = cstn.monte_carlo([schedule, schedule_to_compare])
#             result["independent mc probability"] = mc_probabilities[1]
#             result["correlated mc probability"] = mc_probabilities[0]
#             results.append(result)


for coeff in correlation_coefficients:
    for deadline in deadline_factors:
        for item1 in deadline_move:
            result = {}
            # # Makes timepoints and constraints.
            b0 = TimePoint(0, "Begin Travel 1")
            b1 = TimePoint(1, "End Travel 1")
            b2 = TimePoint(2, "Begin Travel 2")
            b3 = TimePoint(3, "End Travel 2")
            c1 = ProbabilisticConstraint(b0, b1, "Travel 1", {"mean": 60, "sd": sd1})
            c2 = Constraint(b1, b2, "Collect", {"lb": 0, "ub": inf})
            c3 = ProbabilisticConstraint(b2, b3, "Travel 2", {"mean": 100, "sd": sd2})
            c4 = Constraint(b0, b3, "Deadline", {"lb": ub * (1 - deadline) + item1, "ub": ub * deadline + item1})

            # Makes toy stn from paper while varying parameters
            cstn = CorrelatedTemporalNetwork()
            cstn.time_points = [b0, b1, b2, b3]
            cstn.constraints = [c1, c2, c3, c4]
            if coeff != 0:
                corr = Correlation([c1, c3])
                corr.add_correlation(np.array([[1, coeff],[coeff, 1]]))
                cstn.add_correlation(corr)

            cstn.name = "toy_corr_{}_deadline_{}_sd_{}".format(("").join(str(round(coeff, 2)).split(".")), ("").join(str(round(deadline, 2)).split(".")), ("").join(str(round(item1, 2)).split(".")))
            cstn.save_as_json("temporal-planning-domains/toy/networks/{}".format(cstn.name))
            result["name"] = cstn.name
            result["correlation coefficient"] = coeff
            result["deadline factor"] = deadline
            result["deadline move"] = item1
            for item in indpendent_results:
                if item["deadline factor"] == deadline and item["deadline move"] == item1:
                    result["independent probability"] = item["probability"]
                    result["independent delta"] = item["delta"]
                    result["independent runtime"] = item["runtime"]
                    schedule_to_compare = item["schedule"]

            # Solves using column generation and gets schedule
            op = PstnOptimisation(cstn, verbose=True)
            start_t = time()
            op.optimise()
            runtime = time() - start_t
            convex = op.solutions[-1]
            schedule = convex.get_schedule()
            result["correlated delta"] = schedule["2"] - schedule["0"]
            result["correlated probability"] = convex.get_probability()
            result["correlated runtime"] = runtime
            # Gets monte-carlo probability
            mc_probabilities = cstn.monte_carlo([schedule, schedule_to_compare])
            result["independent mc probability"] = mc_probabilities[1]
            result["correlated mc probability"] = mc_probabilities[0]
            results.append(result)

keys = results[0].keys()
with open("junk/results_toy_scale.csv", "w", newline='') as f:
    writer = csv.DictWriter(f, keys)
    writer.writeheader()
    writer.writerows(results)

