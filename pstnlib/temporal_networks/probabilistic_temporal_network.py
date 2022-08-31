
from pstnlib.temporal_networks.temporal_network import TemporalNetwork
from pstnlib.temporal_networks.constraint import Constraint, ProbabilisticConstraint
from pstnlib.temporal_networks.timepoint import TimePoint
import subprocess
from graphviz import Digraph
import json
inf = 1000000000

class ProbabilisticTemporalNetwork(TemporalNetwork):
    """
    represents a probabilistic temporal network.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def parse_from_temporal_network(self, temporal_network: TemporalNetwork):
        """
        makes a probabilistic temporal network from existing temporal network.
        """
        for node in temporal_network.time_points:
            self.add_time_point(node.copy())

        for constraint in temporal_network.constraints:
            source = self.get_timepoint_by_id(constraint.source.id)
            sink = self.get_timepoint_by_id(constraint.sink.id)
            to_add = Constraint(source, sink, constraint.label[:], constraint.duration_bound.copy())
            self.add_constraint(to_add)
    
    def parse_from_json(self, json_file):
        """
        This function parses a JSON file and returns an instance of the temporal_network class.
        """
        if json_file[-5:] != ".json":
            json_file = json_file + ".json"

        # Opens the json and extracts the nodes and edge data
        with open(json_file) as f:
            data = json.load(f)
    
        nodes, edges = data["timepoints"], data["constraints"]

        # Adds the nodes and edges
        for node in nodes:
            to_add = TimePoint(node["id"], node["label"])
            self.add_time_point(to_add)

        for edge in edges:
            source, sink = self.get_timepoint_by_id(edge["source"]), self.get_timepoint_by_id(edge["sink"])
            if edge["type"] == "stc":
                to_add = Constraint(source, sink, edge["label"], {"lb": edge["duration_bound"]["lb"], "ub": edge["duration_bound"]["ub"]})
            elif edge["type"] == "pstc":
                to_add = ProbabilisticConstraint(source, sink, edge["label"], {"mean": edge["distribution"]["mean"], "sd": edge["distribution"]["sd"]})
            self.add_constraint(to_add)

    
    def parse_uncertainties_from_json(self, file: json):
        """
        Reads in a json of action and til uncertainties, such that the uncertainty x = sd/mean. Updates edges with
        distributions and makes edges probabilistic. Returns a PSTN.
        """
        if file[-5:] != ".json":
            file = file + ".json"
        with open(file) as f:
            uncertainties = json.load(f)
        actions, tils = uncertainties["actions"], uncertainties["tils"]

        for action in actions:
            for i in range(len(self.constraints)):
                if action["name"] in self.constraints[i].label:
                    if self.constraints[i].type == "stc":
                        # Replaces the simple temporal constraint with probabilistic version.
                        distribution = {"mean": self.constraints[i].ub * action["mean_fraction"], "sd": self.constraints[i].ub * action["sd_fraction"]}
                        new_constraint = self.constraints[i].copy_as_probabilistic(distribution)
                        self.constraints[i] = new_constraint
                    else:
                        raise ValueError("Uncertainties already added to costraints.")
        for til in tils:
            for i in range(len(self.constraints)):
                if til["name"] in self.constraints[i].label:
                    if self.constraints[i].type == "stc":
                        # Replaces the simple temporal constraint with probabilistic version.
                        distribution = {"mean": self.constraints[i].ub * action["mean_fraction"], "sd": self.constraints[i].ub * action["sd_fraction"]}
                        new_constraint = self.constraints[i].copy_as_probabilistic(distribution)
                        self.constraints[i] = new_constraint
                    else:
                        raise ValueError("Uncertainties already added to constraints.")

    def add_constraint(self, constraint: Constraint) -> None:
        """
        add an edge (constraint) to the network. permits constraints of type 'pstc'.
        """
        # Checks if there is already a constraint with those time-points, if not it adds
        existing = self.get_constraint_by_timepoint(constraint.source, constraint.sink)
        if existing == None:
            self.constraints.append(constraint)
            if constraint.source not in self.time_points:
                self.add_time_point(constraint.source)
            if constraint.sink not in self.time_points:
                self.add_time_point(constraint.sink)
        # If the source and sink time-points are the same way round in the new constraint versus existing
        elif existing.source == constraint.source:
            if existing.type == "stc" and constraint.type == "stc":
                # Checks whether the new constraint has a tighter bound
                if constraint.ub < existing.ub:
                    existing.duration_bound["ub"] = constraint.ub
                elif constraint.lb > existing.lb:
                    existing.duration_bound["lb"] = constraint.lb
            elif existing.type == "stc" and constraint.type == "pstc":
                # Replaces the constraint with probabilistic version.
                existing_index = self.constraints.index(existing)
                self.constraints[existing_index] = constraint
            elif existing.type == "pstc" and constraint.type == "stc":
                # Replaces the probabilistic constraint with normal version.
                existing_index = self.constraints.index(existing)
                self.constraints[existing_index] = constraint
        # If the source and sink time-points are the wrong way round in the new constraint versus existing
        elif existing.sink == constraint.source:
            if existing.type == "stc" and constraint.type == "stc":
                if -constraint.lb < existing.ub:
                    existing.duration_bound["ub"] = -constraint.lb
                if -constraint.ub > existing.lb:
                    existing.duration_bound["lb"] = -constraint.ub
            elif existing.type == "stc" and constraint.type == "pstc":
                # Replaces the constraint with probabilistic version.
                existing_index = self.constraints.index(existing)
                self.constraints[existing_index] = constraint
            elif existing.type == "pstc" and constraint.type == "stc":
                # Replaces the probabilistic constraint with normal version.
                existing_index = self.constraints.index(existing)
                self.constraints[existing_index] = constraint
        
    def get_probabilistic_constraints(self) -> list[Constraint]:
        """
        returns a list of probabilistic constraints (those with type = pstc)
        """
        return [i for i in self.constraints if i.type == "pstc"]

    def get_requirement_constraints(self) -> list[Constraint]:
        """
        returns a list of requirement constraints (those with type = stc)
        """
        return [i for i in self.constraints if i.type == "stc"]

    def set_controllability_of_time_points(self) -> None:
        """
        checks which time_points are uncontrollable (i.e. they come at the end of a probabilistic constraint). Sets the controllable flag
        to True if controllable and False if not controllable
        """
        uncontrollable_time_points = [i.sink for i in self.get_probabilistic_constraints()]
        for time_point in self.time_points:
            if time_point in uncontrollable_time_points:
                time_point.controllable = False
            else:
                time_point.controllable = True
    
    def get_controllable_time_points(self) -> list[TimePoint]:
        """
        returns a list of controllable time-points
        """
        self.set_controllability_of_time_points()
        return [i for i in self.time_points if i.controllable == True]
    
    def get_uncontrollable_time_points(self) -> list[TimePoint]:
        """
        returns a list of uncontrollable time-points
        """
        self.set_controllability_of_time_points()
        return [i for i in self.time_points if i.controllable == False]
    
    def get_uncontrollable_constraints(self) -> list[Constraint]:
        """
        returns a list of requirement constraints that contain an uncontrollable time-point
        """
        self.set_controllability_of_time_points()
        uncontrollable_constraints = []
        for constraint in self.constraints:
            if constraint.source.controllable == False or constraint.sink.controllable == False:
                uncontrollable_constraints.append(constraint)
        return uncontrollable_constraints
    
    def get_controllable_constraints(self) -> list[Constraint]:
        self.set_controllability_of_time_points()
        return [i for i in self.get_requirement_constraints() if i.source.controllable == True and i.sink.controllable == True]

    def get_outgoing_uncontrollable_edge_from_timepoint(self, timepoint: TimePoint) -> list[Constraint]:
        """
        given a time-point i, returns a list of all outgoing edges (i, j)
        """
        return [ij for ij in self.constraints if ij.source == timepoint and ij.type == "stc"]

    def get_incoming_uncontrollable_edge_from_timepoint(self, timepoint: TimePoint) -> list[Constraint]:
        """
        given a time-point j, returns a list of all incoming edges (i, j)
        """
        return [ij for ij in self.constraints if ij.sink == timepoint and ij.type == "stc"]
    
    def plot_dot_graph(self):
        """
        Plot the graph as a dot graph using graphviz
        """
        requirements = self.get_requirement_constraints()
        probabilistics = self.get_probabilistic_constraints()

        plot = Digraph()
        for timePoint in self.time_points:
            plot.node(name=str(timePoint.id), label=str(timePoint.id))
        
        for constraint in requirements:
            plot.edge(str(constraint.source.id), str(constraint.sink.id), label="{}: [{}, {}]".format(constraint.label, constraint.lb, constraint.ub))
        for constraint in probabilistics:
            plot.edge(str(constraint.source.id), str(constraint.sink.id), label="{}: N({}, {})".format(constraint.label, constraint.mean, constraint.sd))
        try:
            plot.render('junk/{}.png'.format(self.name), view=True)
        except subprocess.CalledProcessError:
            print("Please close the PDF and rerun the script")

