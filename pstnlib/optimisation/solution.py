import numpy as np
from pstnlib.temporal_networks.constraint import ProbabilisticConstraint
from pstnlib.temporal_networks.probabilistic_temporal_network import ProbabilisticTemporalNetwork
from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
from math import exp

class Solution(object):
    """
    Description:    Class representing a solution to PSTN SC as a Gurobi model.
    Parameters:     model - Gurobi model instance containing solution.
                    runtime - cumulative runtime of current iteration.
    """
    def __init__(self, network, model, runtime) -> None:
        self.network = network
        self.model = model
        self.runtime = runtime
    
    def get_probability(self) -> float:
        """
        returns the empirical probability of success from the gurobi model.
        """
        if "RMP" in self.model.getAttr("ModelName"):
            return exp(-self.model.objVal)
        elif "PARIS" in self.model.getAttr("ModelName"):
            # Gets values of probability variables and computes joint outcome
            prob = 1
            for constraint in self.network.get_probabilistic_constraints():
                lower, upper = self.model.getVarByName(constraint.get_description() + "_Fl"), self.model.getVarByName(constraint.get_description() + "_Fu")
                prob *= (1 - (lower.x + upper.x))
            return prob

    def get_schedule(self) -> dict:
        """
        returns the schedule from the gurobi model solution.
        """
        schedule = {}
        for tp in self.network.get_controllable_time_points():
            value = self.model.getVarByName(str(tp.id)).x
            schedule[tp.id] = value
        return schedule

    def simulate_execution(self) -> bool:
        '''
        Description: For a given schedule and PSTN, simulates execution of schedule once and returns True if successful (all constraints met)
                    else returns False.
        
        Input:      PSTN:           instance of PSTN class to be simulated
                    schedule:       dictionary {timepoint0: time,...,timepointn: value} of time-point: time pairs
        
        Output:     bool:           True if successfully executed else False
        '''
        schedule = self.get_schedule()

        if isinstance(self.network, CorrelatedTemporalNetwork):
            for constraint in self.network.get_independent_probabilistic_constraints():
                schedule[constraint.sink.id] = schedule[constraint.source.id] + np.random.normal(constraint.mean, constraint.sd)
            for correlation in self.network.correlations:
                sample = np.random.multivariate_normal(correlation.mean, correlation.covariance)
                for i in range(len(correlation.constraints)):
                    constraint = correlation.constraints[i]
                    schedule[constraint.sink.id] = schedule[constraint.source.id] + sample[i]
        elif isinstance(self.network, ProbabilisticTemporalNetwork):
            for constraint in self.network.get_probabilistic_constraints():
                schedule[constraint.sink.id] = schedule[constraint.source.id] + np.random.normal(constraint.mean, constraint.sd)
        else:
            raise TypeError("Network must be ProbabilisticTemporalNetwork or CorrelatedTemporalNetwork")

        for constraint in self.network.constraints:
            if not isinstance(constraint, ProbabilisticConstraint):
                start, end = schedule[constraint.source.id], schedule[constraint.sink.id]
                if round(end - start, 10) < round(constraint.lb, 10) or round(end - start, 10) > round(constraint.ub, 10):
                    return False
        return True

    def monte_carlo(self, no_simulations: int = 10000) -> float:
        '''
        Description:    Simulates execution of schedule a set amount of times and return probability
                        of successfule execution (i.e. all constraints satisfied)
        
        Input:      PSTN:           instance of PSTN class to be simulated
                    schedule:       dictionary {timepoint0: time,...,timepointn: value} of time-point: time pairs
                    no_simulations: number of times to simulate execution
        
        Output:     float:          no times successfully executed/total number of simulations
        '''
        count = 0
        for i in range(no_simulations):
            if self.simulate_execution() == True:
                count += 1
        return count/no_simulations
