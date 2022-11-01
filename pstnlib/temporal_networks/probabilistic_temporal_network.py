from json.encoder import INFINITY
from pstnlib.temporal_networks.temporal_network import TemporalNetwork
from pstnlib.temporal_networks.constraint import Constraint, ProbabilisticConstraint
from pstnlib.temporal_networks.timepoint import TimePoint
import subprocess
from graphviz import Digraph
import json
import numpy as np
inf = 1e9

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

    def parse_uncertainties_from_dict(self, uncertainties: dict):
        """
        Reads in a json of action and til uncertainties, such that the uncertainty x = sd/mean. Updates edges with
        distributions and makes edges probabilistic. Returns a PSTN.
        """
        actions, tils = uncertainties["actions"], uncertainties["tils"]

        for action in actions:
            for i in range(len(self.constraints)):
                if action["name"] in self.constraints[i].label:
                    if self.constraints[i].type == "stc":
                        # Replaces the simple temporal constraint with probabilistic version.
                        distribution = {"mean": self.constraints[i].ub * action["mean_fraction"], "sd": self.constraints[i].ub * action["sd_fraction"]}
                        new_constraint = self.constraints[i].copy_as_probabilistic(distribution)
                        self.constraints[i] = new_constraint

        for til in tils:
            for i in range(len(self.constraints)):
                if til["name"] in self.constraints[i].label:
                    if self.constraints[i].type == "stc":
                        # Replaces the simple temporal constraint with probabilistic version.
                        distribution = {"mean": self.constraints[i].ub * action["mean_fraction"], "sd": self.constraints[i].ub * action["sd_fraction"]}
                        new_constraint = self.constraints[i].copy_as_probabilistic(distribution)
                        self.constraints[i] = new_constraint
    
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
                    # else:
                    #     raise ValueError("Uncertainties already added to costraints.")
        for til in tils:
            for i in range(len(self.constraints)):
                if til["name"] in self.constraints[i].label:
                    if self.constraints[i].type == "stc":
                        # Replaces the simple temporal constraint with probabilistic version.
                        distribution = {"mean": self.constraints[i].ub * action["mean_fraction"], "sd": self.constraints[i].ub * action["sd_fraction"]}
                        new_constraint = self.constraints[i].copy_as_probabilistic(distribution)
                        self.constraints[i] = new_constraint
                    # else:
                    #     raise ValueError("Uncertainties already added to constraints.")

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
            if (constraint.source.controllable == False or constraint.sink.controllable == False) and not isinstance(constraint, ProbabilisticConstraint):
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
    
    def get_incoming_probabilistic(self, constraint: Constraint) -> dict:
        """
        given a simple temporal constraint returns dictionary of incoming probabilistic constraints in the form:
        {"start": incoming, "end": incoming}, where if constraint is edge (i, j), start is the constraint (k, i) 
        and end is the constraint (k, j). If no incoming probabilistic at either, the value is set to None 
        """
        self.set_controllability_of_time_points()
        if constraint.source.controllable == True and constraint.sink.controllable == True:
            return {"start": None, "end": None}
        else:
            incoming_source = [g for g in self.get_probabilistic_constraints() if g.sink == constraint.source]
            incoming_sink = [g for g in self.get_probabilistic_constraints() if g.sink == constraint.sink]
            if len(incoming_source) > 1 or len(incoming_sink) > 1:
                raise AttributeError("More than one incoming contingent edge")
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
            plot.edge(str(constraint.source.id), str(constraint.sink.id), label="{}: N({}, {})".format(constraint.label, constraint.mean, constraint.sd), color='red')
        try:
            plot.render('junk/{}.png'.format(self.name), view=True)
        except subprocess.CalledProcessError:
            print("Please close the PDF and rerun the script")
            
    def simulate_execution(self, schedules) -> bool:
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
            # samples outcome for probabilistic constraints and adds to each schedule.
            for constraint in self.get_probabilistic_constraints():
                sample = np.random.normal(constraint.mean, constraint.sd)
                for schedule in schedules:
                    schedule[str(constraint.sink.id)] = schedule[str(constraint.source.id)] + sample

            toReturn = [True for i in range(len(schedules))]
            # for each schedule finds out if any constraints are violated and if so saves False to return list, else saves True.
            for schedule in schedules:
                for constraint in self.constraints:
                    if not isinstance(constraint, ProbabilisticConstraint):
                        start, end = schedule[str(constraint.source.id)], schedule[str(constraint.sink.id)]
                        if round(end - start, 10) < round(constraint.lb, 10) or round(end - start, 10) > round(constraint.ub, 10):
                            toReturn[schedules.index(schedule)] = False
            return toReturn

        # Otherwise it only tests one and returns bool.
        elif isinstance(schedules, dict):
            for constraint in self.get_independent_probabilistic_constraints():
                sample = np.random.normal(constraint.mean, constraint.sd)
                schedule[str(constraint.sink.id)] = schedule[str(constraint.source.id)] + sample
            # Finds out if any of the constraints are violated, if so returns False, else returns True  
            for constraint in self.constraints:
                if not isinstance(constraint, ProbabilisticConstraint):
                    start, end = schedule[str(constraint.source.id)], schedule[str(constraint.sink.id)]
                    if round(end - start, 10) < round(constraint.lb, 10) or round(end - start, 10) > round(constraint.ub, 10):
                        return False
            return True
        else:
            raise ValueError("Input parameter schedules must be either a list of dictionaries or a single dictionary.")

    def monte_carlo(self, schedules, no_simulations: int = 10000) -> float:
        '''
        Description:    Simulates execution of schedules a set amount of times and return probability
                        of successful execution (i.e. all constraints satisfied)
        
        Input:          schedules:      A schedule is a dictionary {timepoint0: time,...,timepointn: value} of time-point: time pairs.
                                        Multiple schedules can be passed as a list.
                        no_simulations: number of times to simulate execution
        
        Output:         float:          probability of success: no times successfully executed/total number of simulations
        '''
        # If a list of schedules is input.
        if isinstance(schedules, list):
            counts = [0 for i in range(len(schedules))]
            for i in range(no_simulations):
                result = self.simulate_execution(schedules)
                for j in range(len(schedules)):
                    if result[j] == True:
                        counts[j] += 1
            probs = [counts[i]/no_simulations for i in range(len(counts))]
            return probs

        # If a single schedule is input.
        elif isinstance(schedules, dict):
            count = 0
            for i in range(no_simulations):
                result = self.simulate_execution(schedules)
                if result == True:
                    count += 1
            return count/no_simulations
        else:
            raise ValueError("Input parameter schedules must be either a list of dictionaries or a single dictionary.")

