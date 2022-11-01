from pstnlib.temporal_networks.probabilistic_temporal_network import ProbabilisticTemporalNetwork
from pstnlib.temporal_networks.correlation import Correlation
from pstnlib.temporal_networks.constraint import Constraint, ProbabilisticConstraint
import json
import numpy as np
from pstnlib.temporal_networks.numpy_encoder import NpEncoder

np.set_printoptions(linewidth=160)
class CorrelatedTemporalNetwork(ProbabilisticTemporalNetwork):
    """
    represents a correlated probabilistic temporal network.
    """
    def __init__(self) -> None:
        super().__init__()
        self.correlations = []
    
    def parse_from_probabilistic_temporal_network(self, network: ProbabilisticTemporalNetwork):
        """
        makes a correlated temporal network from existing probabilistic temporal network.
        """
        for node in network.time_points:
            self.add_time_point(node)

        for constraint in network.constraints:
            self.add_constraint(constraint)
    
    def copy(self):
        copied_network = CorrelatedTemporalNetwork()
        copied_timepoints = [t.copy() for t in self.time_points]
        copied_constraints = []
        # Copies the constraints.
        for constraint in self.constraints:
            for tp in copied_timepoints:
                if tp.id == constraint.source.id:
                    source = tp
                elif tp.id == constraint.sink.id:
                    sink = tp
            if isinstance(constraint, ProbabilisticConstraint):
                copied_constraints.append(ProbabilisticConstraint(source, sink, constraint.label[:], constraint.distribution.copy()))
            else:
                copied_constraints.append(Constraint(source, sink, constraint.label[:], constraint.duration_bound.copy()))
        # Copies the correlations
        copied_correlations = []
        if self.correlations:
            for corr in self.correlations:
                correlated_constraints = [c for c in copied_constraints if c.get_description() in [c.get_description() for c in corr.constraints]]
                correlation_matrix = corr.correlation
                copied_correlation = Correlation(correlated_constraints)
                copied_correlation.add_correlation(correlation_matrix)
                copied_correlations.append(copied_correlation)
        # Adds all of the copied data and returns
        copied_network.time_points = copied_timepoints
        copied_network.constraints = copied_constraints
        copied_network.correlations = copied_correlations
        if self.name != None:
            copied_network.name = self.name[:]
        return copied_network
                

    def parse_from_json(self, json_file):
        """
        This function parses a JSON file and returns an instance of the temporal_network class.
        """
        super().parse_from_json(json_file)

        if json_file[-5:] != ".json":
            json_file = json_file + ".json"

        with open(json_file) as f:
            data = json.load(f)
            
        self.name = data["name"]

        if "correlations" in data:
            correlations = data["correlations"]
            for correlation in correlations:
                constraints = []
                for constraint in correlation["constraints"]:
                    constraints.append(self.get_constraint_by_timepoint(self.get_timepoint_by_id(constraint["source"]), self.get_timepoint_by_id(constraint["sink"])))
                toAdd = Correlation(constraints)
                toAdd.add_correlation(np.array(correlation["correlation"]))
                self.add_correlation(toAdd)

    def add_correlation(self, correlation: Correlation) -> None:
        """
        Adds an instance of correlation
        """
        self.correlations.append(correlation)
    
    def get_correlated_probabilistic_constraints(self) -> list[Constraint]:
        """
        returns list of all probabilistic constraints involved in a correlation
        """
        to_return = []
        for correlation in self.correlations:
            for constraint in correlation.constraints:
                if constraint not in to_return:
                    to_return.append(constraint)
        return to_return

    def get_independent_probabilistic_constraints(self) -> list[Constraint]:
        """
        returns list of all probabilistic constraints not incvolved in a correlation
        """
        to_return = []
        correlated = self.get_correlated_probabilistic_constraints()
        for constraint in self.get_probabilistic_constraints():
            if constraint not in correlated:
                to_return.append(constraint)
        return to_return

    def print_as_json(self):
        """
        print the graph in JSON format.
        """
        print("{")
        print("\t\"timepoints\": [")
        for time_point in self.time_points:
            print("\t\t{\"id\": " + str(time_point.id) + ", \"label\": \"" + time_point.label + "\"},")
        print("\t],")
        print("\t\"constraints\": [")
        for constraint in self.constraints:
            if constraint.type == "stc":
                print("\t\t{\"source\": " + str(constraint.source.id) + ", \"target\": " + str(constraint.sink.id) + ", \"label\": \"" + constraint.label + "\", \"bounds\": " + "({}, {})".format(constraint.lb, constraint.ub) + "},")
            elif constraint.type == "pstc":
                print("\t\t{\"source\": " + str(constraint.source.id) + ", \"target\": " + str(constraint.sink.id) + ", \"label\": \"" + constraint.label + "\", \"distribution\": " + "N({}, {})".format(constraint.mean, constraint.sd) + "},")
        print("\t]")
        print("\t\"correlations\": [")
        for correlation in self.correlations:
            print("\t\t{\"constraints\": " + str([c.get_description() for c in correlation.constraints]) + ", \"mean\": " + str(correlation.mean) + ", \"correlation\": " + str([list(i) for i in correlation.correlation]) + "},")
        print("\t]")
        print("}")
    
    def save_as_json(self, filename):
        """
        saves the network as a JSON to filename.json
        """
        if filename[-5:] != ".json":
            filename = filename + ".json"
        toDump = {"name": self.name}
        toDump["timepoints"] = [t.to_json() for t in self.time_points]
        toDump["constraints"] = [c.to_json() for c in self.constraints]
        toDump["correlations"] = [c.to_json() for c in self.correlations]
        with open(filename, 'w') as fp:
            json.dump(toDump, fp, indent=4, separators=(", ", ": "), cls = NpEncoder)
    
    def get_json(self):
        toReturn = {"name": self.name}
        toReturn["timepoints"] = [t.to_json() for t in self.time_points]
        toReturn["constraints"] = [c.to_json() for c in self.constraints]
        toReturn["correlations"] = [c.to_json() for c in self.correlations]
        return toReturn

    def simulate_execution(self, schedules):
        '''
        Description: For a given schedule (or list of schedules) and PSTN, simulates execution of schedule once and returns True if successful (all constraints met)
                    else returns False.
        
        Input:      PSTN:          Instance of PSTN class to be simulated
                    schedules:     Schedules to simulate. Each schedule is a dictionary of timepoint id: value. If a list of schedules is passed it will test each.
        Output:     bool:          True if successfully executed else False. returns a list of bools if a list of schedules is passed.
        '''
        # if multiple schedules are passed it tests all of them and returns a list of bools.
        if isinstance(schedules, list):
            toReturn = []
            # samples outcome for independent probabilistic constraints and adds to each schedule.
            for constraint in self.get_independent_probabilistic_constraints():
                sample = np.random.normal(constraint.mean, constraint.sd)
                for schedule in schedules:
                    schedule[str(constraint.sink.id)] = schedule[str(constraint.source.id)] + sample
            # samples outcomes for correlations and adds to each schedule.
            for correlation in self.correlations:
                sample = np.random.multivariate_normal(correlation.mean, correlation.covariance)
                for schedule in schedules:
                    for i in range(len(correlation.constraints)):
                        constraint = correlation.constraints[i]
                        schedule[str(constraint.sink.id)] = schedule[str(constraint.source.id)] + sample[i]

            toReturn = [True for i in range(len(schedules))]

            for i in range(len(schedules)):
                #print("\nChecking schedule at index: ", i)
                for constraint in self.constraints:
                    if not isinstance(constraint, ProbabilisticConstraint):
                        start, end = schedules[i][str(constraint.source.id)], schedules[i][str(constraint.sink.id)]
                        #print("Source", constraint.source.id, "Sink", constraint.sink.id)
                        #print("Value: ", end - start)
                        if round(end - start, 6) < round(constraint.lb, 6) or round(end - start, 6) > round(constraint.ub, 6):
                            #print("Constraint violated: ", constraint.source.id, constraint.sink.id, constraint.duration_bound)
                            toReturn[i] = False
            #print("\n", toReturn)
            return toReturn

        # Otherwise it only tests one and returns bool.
        elif isinstance(schedules, dict):
            for constraint in self.get_independent_probabilistic_constraints():
                schedules[str(constraint.sink.id)] = schedules[str(constraint.source.id)] + np.random.normal(constraint.mean, constraint.sd)
            for correlation in self.correlations:
                sample = np.random.multivariate_normal(correlation.mean, correlation.covariance)
                for i in range(len(correlation.constraints)):
                    constraint = correlation.constraints[i]
                    schedules[str(constraint.sink.id)] = schedules[str(constraint.source.id)] + sample[i]
            # Finds out if any of the constraints are violated, if so returns False, else returns True  
            for constraint in self.constraints:
                if not isinstance(constraint, ProbabilisticConstraint):
                    start, end = schedules[str(constraint.source.id)], schedules[str(constraint.sink.id)]
                    if round(end - start, 10) < round(constraint.lb, 10) or round(end - start, 10) > round(constraint.ub, 10):
                        return False
            return True
        else:
            raise ValueError("Input parameter schedules must be either a list of dictionaries or a single dictionary.")
    
    def calculate_trace(self) -> float:
        """
        Calculates the trace of the covariance matrix defining the random vector associated with the covariance matrix.
        The trace is normalised by the number of dimensions in the distribution. This is used to measure the "overall"
        variance of the matrix.
        Larger values represent greater overall variance
        """
        # Initialises an n by n matrix where n is the number of probabilistic constraints.
        size = len(self.get_probabilistic_constraints())
        cov = np.zeros((size, size))
        index = 0
        # Adds the covariance matrix associated with each correlated outcome.
        for correlation in self.correlations:
            for row in range(np.shape(correlation.covariance)[0]):
                for col in range(np.shape(correlation.covariance)[1]):
                    cov[index + row, index + col] = correlation.covariance[row, col]
            index += np.shape(correlation.covariance)[0]
        independent_constraints = self.get_independent_probabilistic_constraints()
        # Adds indepentdent probabilistic constraint variances on the diagonal.
        for i in range(len(independent_constraints)):
            cov[index + i, index + i] = independent_constraints[i].sd**2
        return np.trace(cov)/size
    
    def calculate_generalized_variance(self) -> float:
        """
        Calculates the determinant of the correlation matrix. This is used to measure the "overall"
        correlation of the matrix.
        Larger values suggest little correlation, whereas smaller values represent larger correlation.
        """
        # Initialises an n by n matrix where n is the number of probabilistic constraints.
        size = len(self.get_probabilistic_constraints())
        corr = np.zeros((size, size))
        index = 0
        # Adds the covariance matrix associated with each correlated outcome.
        for correlation in self.correlations:
            for row in range(np.shape(correlation.correlation)[0]):
                for col in range(np.shape(correlation.correlation)[1]):
                    corr[index + row, index + col] = correlation.correlation[row, col]
            index += np.shape(correlation.correlation)[0]
        independent_constraints = self.get_independent_probabilistic_constraints()
        # Adds indepentdent probabilistic constraint variances on the diagonal.
        for i in range(len(independent_constraints)):
            corr[index + i, index + i] = 1
        return np.linalg.det(corr)
    
    def get_largest_correlation_coefficient(self) -> float:
        """
        Finds the largest absolute correlation coefficient amongst all the networks correlations.
        """
        largest = 0
        for correlation in self.correlations:
            for i in range(len(correlation.constraints)):
                for j in range(len(correlation.constraints)):
                    if correlation.correlation[i, j] != 1 and abs(correlation.correlation[i, j]) > largest:
                        largest = abs(correlation.correlation[i, j])
        return largest
 
                



