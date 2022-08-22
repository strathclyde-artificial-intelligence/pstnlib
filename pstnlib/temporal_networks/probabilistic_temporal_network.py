
from pstnlib.temporal_networks.temporal_network import TemporalNetwork
from pstnlib.temporal_networks.constraint import Constraint
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
            to_add = Constraint(source, sink, constraint.label[:], constraint.type[:], constraint.duration_bound.copy())
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
            if "distribution" in edge:
                distribution = edge["distribution"]
            else:
                distribution = None
            to_add = Constraint(source, sink, edge["label"], edge["type"], {"lb": edge["duration_bound"]["lb"], "ub": edge["duration_bound"]["ub"]}, distribution)
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
            for constraint in self.constraints:
                if action["name"] in constraint.label:
                    if constraint.type == "stc":
                        assert constraint.ub == constraint.lb
                        constraint.distribution = {"mean": constraint.ub * action["mean_fraction"], "sd": constraint.ub * action["sd_fraction"]}
                        constraint.duration_bound["lb"], constraint.duration_bound["ub"] = 0, inf
                        constraint.type = "pstc"
                    else:
                        raise ValueError("Uncertainties already added to costraints.")
        for til in tils:
            for constraint in self.constraints:
                if til["name"] in constraint.label:
                    if constraint.type == "stc":
                        assert constraint.ub == constraint.lb
                        constraint.distribution = {"mean": constraint.ub * til["mean_fraction"], "sd": constraint.ub * til["sd_fraction"]}
                        constraint.duration_bound["lb"], constraint.duration_bound["ub"] = 0, inf
                        constraint.type = "pstc"
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
            # Checks whether the new constraint has a tighter bound
            if constraint.ub < existing.ub:
                existing.ub = constraint.ub
            elif constraint.lb > existing.lb:
                existing.lb = constraint.lb
        # If the source and sink time-points are the wrong way round in the new constraint versus existing
        elif existing.sink == constraint.source:
            if -constraint.lb < existing.ub:
                existing.ub = -constraint.lb
            if -constraint.ub > existing.lb:
                existing.lb = -constraint.ub
    
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
    
    def incoming_probabilistic(self, constraint: Constraint) -> dict[str, Constraint]:
        """
        returns a dictionary of the incoming probabilistic constraint in the form {"start": Constraint, "end": Constraint}
        raises an exception if the number of incoming probabilistic constraints is greater than one
        """
        if constraint not in self.get_uncontrollable_constraints():
            return None
        else:
            incoming_source = [g for g in self.getContingents() if g.sink == constraint.source]
            incoming_sink = [g for g in self.getContingents() if g.sink == constraint.sink]
            if len(incoming_source) > 1 or len(incoming_sink) > 1:
                raise AttributeError("More than one incoming probabilistic edge.")
            else:
                try:
                    return {"start": incoming_source[0], "end": incoming_sink[0]}
                except IndexError:
                    try:
                        return {"start": incoming_source[0], "end": None}
                    except IndexError:
                        return {"start": None, "end": incoming_sink[0]}
    
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
            plot.render('logs/{}.png'.format(self.name), view=True)
        except subprocess.CalledProcessError:
            print("Please close the PDF and rerun the script")