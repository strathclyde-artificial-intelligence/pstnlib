from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
from pstnlib.temporal_networks.timepoint import TimePoint
from pstnlib.temporal_networks.constraint import Constraint, ProbabilisticConstraint
from pstnlib.optimisation.column_generation import PstnOptimisation
from pstnlib.temporal_networks.correlation import Correlation
import numpy as np
import csv
from time import time

inf = 1e9

# Makes timepoints.
b0 = TimePoint(0, "Begin Travel 1")
b1 = TimePoint(1, "End Travel 1")
b2 = TimePoint(2, "Begin Travel 2")
b3 = TimePoint(3, "End Travel 2")

# Makes constraints.
c1 = ProbabilisticConstraint(b0, b1, "Travel 1", {"mean": 60, "sd": 10})
c2 = Constraint(b1, b2, "Collect", {"lb": 0, "ub": inf})
c3 = ProbabilisticConstraint(b2, b3, "Travel 2", {"mean": 100, "sd": 25})
c4 = Constraint(b0, b3, "Deadline", {"lb": 0, "ub": 160})

# Makes a correlation.
corr = Correlation([c1, c3])
corr.add_correlation(np.array([[1, 0.9], [0.9, 1]]))

# Makes Correlated Temporal Network.
cstn = CorrelatedTemporalNetwork()
cstn.time_points = [b0, b1, b2, b3]
cstn.constraints = [c1, c2, c3, c4]
cstn.add_correlation(corr)
cstn.name = "toy_cstn"

# Solves considering the correlation.
corr_prob = PstnOptimisation(cstn, verbose=1)
corr_prob.optimise()
print("\nProbability with correlation", corr_prob.solutions[-1].get_probability())

# Solves assuming independence.
ind_prob = PstnOptimisation(cstn, verbose=1, assume_independence=True)
ind_prob.optimise()
print("Probability assuming independence", ind_prob.solutions[-1].get_probability())

# Gets the schedules.
corr_sched = corr_prob.solutions[-1].get_schedule()
ind_sched = ind_prob.solutions[-1].get_schedule()
print("\nSchedule with Correlation", corr_sched)
print("Schedule with independence", ind_sched)
schedules = [corr_sched, ind_sched]

# Simulates execution of schedules.
mc_probs = cstn.monte_carlo(schedules)
print("\nMonte Carlo probability with correlation", mc_probs[0])
print("Monte Carlo probability assuming independence", mc_probs[1])
