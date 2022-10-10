from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
from pstnlib.temporal_networks.timepoint import TimePoint
from pstnlib.temporal_networks.constraint import Constraint, ProbabilisticConstraint
from pstnlib.optimisation.pstn_optimisation_class import PstnOptimisation
from pstnlib.temporal_networks.correlation import Correlation
import numpy as np
import csv

inf = 1e9

sd1 = 10
sd2 = 25
ub = 200

results = []
# correlation_coefficients = np.array([0, 0.3, 0.6, 0.9])
# deadline_factors = np.array([0.6, 0.7, 0.8, 0.9, 1.0,])
# sd_factors = np.array([0.6, 0.8, 1, 1.2, 1.4])
correlation_coefficients = np.array([0.9])
deadline_factors = np.array([1])
sd_factors = np.array([1.2])
for coeff in correlation_coefficients:
    for deadline in deadline_factors:
        for sd in sd_factors:
            result = {}
            # # Makes timepoints and constraints.
            b0 = TimePoint(0, "Begin Travel 1")
            b1 = TimePoint(1, "End Travel 1")
            b2 = TimePoint(2, "Begin Travel 2")
            b3 = TimePoint(3, "End Travel 2")
            c1 = ProbabilisticConstraint(b0, b1, "Travel 1", {"mean": 60, "sd": sd1 * sd})
            c2 = Constraint(b1, b2, "Collect", {"lb": 0, "ub": inf})
            c3 = ProbabilisticConstraint(b2, b3, "Travel 2", {"mean": 100, "sd": sd2 * sd})
            c4 = Constraint(b0, b3, "Deadline", {"lb": ub * (1 - deadline), "ub": ub * deadline})

            # Makes toy stn from paper while varying parameters
            cstn = CorrelatedTemporalNetwork()
            cstn.time_points = [b0, b1, b2, b3]
            cstn.constraints = [c1, c2, c3, c4]
            if coeff != 0:
                corr = Correlation([c1, c3])
                corr.add_correlation(np.array([[1, coeff],[coeff, 1]]))
                cstn.add_correlation(corr)

            cstn.name = "toy_corr_{}_deadline_{}_sd_{}".format(("").join(str(round(coeff, 2)).split(".")), ("").join(str(round(deadline, 2)).split(".")), ("").join(str(round(sd, 2)).split(".")))
            cstn.save_as_json("temporal-planning-domains/toy/networks/{}".format(cstn.name))
            result["name"] = cstn.name
            result["correlation coefficient"] = coeff
            result["deadline factor"] = deadline
            result["sd"] = sd

            # Solves using column generation and gets schedule
            op = PstnOptimisation(cstn, verbose=True)
            op.optimise()
            convex = op.solutions[-1]
            schedule = convex.get_schedule()
            result["delta"] = schedule["2"] - schedule["0"]
            result["probability"] = convex.get_probability()
            # Gets monte-carlo probability
            result["mc probability"] = cstn.monte_carlo(schedule)
            results.append(result)

keys = results[0].keys()
with open("junk/results_toy.csv", "w", newline='') as f:
    writer = csv.DictWriter(f, keys)
    writer.writeheader()
    writer.writerows(results)

