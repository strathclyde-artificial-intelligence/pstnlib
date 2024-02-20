from math import exp
import json
from gurobipy import GRB

class Solution(object):
    """
    Class representing a solution to PSTN SC as a Gurobi model.
    --------------------
    Parameters:
        network:                    pstnlib.temporal_networks.probabilistic_temporal_network.ProbabilisticTemporalNetwork
                                        PSTN/Corr-STN instance solved.
        model:                      gurobipy.Model
                                        Gurobi model for SC
        runtime:                    float
                                        Solution runtime
        experimental_probability:    float
                                        robustness obtained through Monte-Carlo simulation.           
    """
    def __init__(self, network, model, runtime, bound = None) -> None:
        self.network = network
        self.model = model
        self.runtime = runtime
        self.experimental_probability = None
        self.bound = bound
    
    def get_probability(self) -> float:
        """
        returns the empirical probability of success from the gurobi model.
        """
        if "RMP" in self.model.getAttr("ModelName") and self.model.status == GRB.OPTIMAL:
            return exp(-self.model.objVal)
        elif "PARIS" in self.model.getAttr("ModelName") and self.model.status == GRB.OPTIMAL:
            # Gets values of probability variables and computes joint outcome
            prob = 1
            for constraint in self.network.get_probabilistic_constraints():
                lower, upper = self.model.getVarByName(constraint.get_description() + "_Fl"), self.model.getVarByName(constraint.get_description() + "_Fu")
                prob *= (1 - (lower.x + upper.x))
            return prob
        else:
            return None

    def get_schedule(self) -> dict:
        """
        returns the schedule from the gurobi model solution.
        """
        if self.model.status == GRB.OPTIMAL:
            schedule = {}
            for tp in self.network.get_controllable_time_points():
                value = self.model.getVarByName(str(tp.id)).x
                schedule[str(tp.id)] = value
            return schedule
        else:
            return None
    
    def to_json(self, path):
        """
        saves the solution as a json.
        """
        filename = self.model.getAttr("ModelName") + ".json"
        toDump = {}
        toDump["runtime"] = self.runtime
        toDump["schedule"] = self.get_schedule()
        toDump["probability"] = self.get_probability()
        toDump["network"] = self.network.get_json()
        with open(path + "/" + filename, 'w') as fp:
            json.dump(toDump, fp, indent=4, separators=(", ", ": "))
