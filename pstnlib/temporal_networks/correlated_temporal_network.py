from pstnlib.temporal_networks.probabilistic_temporal_network import ProbabilisticTemporalNetwork
from pstnlib.temporal_networks.correlation import Correlation
from pstnlib.temporal_networks.constraint import Constraint, ProbabilisticConstraint
import json
import numpy as np

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
            self.add_time_point(node.copy())

        for constraint in network.constraints:
            source = self.get_timepoint_by_id(constraint.source.id)
            sink = self.get_timepoint_by_id(constraint.sink.id)
            if constraint.type == "stc":
                to_add = Constraint(source, sink, constraint.label[:], constraint.duration_bound.copy())
            else:
                to_add = ProbabilisticConstraint(source, sink, constraint.label[:], constraint.distribution.copy())
            self.add_constraint(to_add)
    
    def parse_from_json(self, json_file):
        """
        This function parses a JSON file and returns an instance of the temporal_network class.
        """
        super().parse_from_json(json_file)

        if json_file[-5:] != ".json":
            json_file = json_file + ".json"

        with open(json_file) as f:
            data = json.load(f)

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
        toDump = {}
        toDump["timepoints"] = [t.to_json() for t in self.time_points]
        toDump["constraints"] = [c.to_json() for c in self.constraints]
        toDump["correlations"] = [c.to_json() for c in self.correlations]
        with open(filename, 'w') as fp:
            json.dump(toDump, fp, indent=4, separators=(", ", ": "))