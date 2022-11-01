from pstnlib.temporal_networks.timepoint import TimePoint
from pstnlib.temporal_networks.constraint import Constraint
from otpl.temporal_networks.simple_temporal_network import SimpleTemporalNetwork
import copy
from queue import PriorityQueue
import json
import subprocess
from graphviz import Digraph
from pstnlib.temporal_networks.numpy_encoder import NpEncoder

inf = 1e9

class TemporalNetwork:
    """
    represents a simple temporal network as a graph.
    """
    def __init__(self) -> None:
        self.name = None
        self.time_points : list[TimePoint] = []
        self.constraints: list[Constraint] = []
    
    def parse_from_temporal_plan_network(self, temporal_plan_network: SimpleTemporalNetwork):
        """
        Parses from an instance of TemporalPlanNetwork as output from the temporal plan. Changes bidirectional edges to be uni-directional.
        """
        for node in temporal_plan_network.nodes:
            self.add_time_point(TimePoint(node, temporal_plan_network.labels[node]))

        for node1 in temporal_plan_network.edges:
            for node2 in temporal_plan_network.edges[node1]:
                if node1 == node2: continue
                elif temporal_plan_network.edges[node1][node2] > 0:
                    # Constraint is an upper bound, adding edge node1 -> node2
                    edge = Constraint(self.get_timepoint_by_id(node1), self.get_timepoint_by_id(node2), temporal_plan_network.edge_labels[node1][node2], {"lb": 0.001, "ub": temporal_plan_network.edges[node1][node2]})
                elif temporal_plan_network.edges[node1][node2] < 0:
                    # Constraint is a lower bound, adding edge node2 -> node1
                    edge = Constraint(self.get_timepoint_by_id(node2), self.get_timepoint_by_id(node1), temporal_plan_network.edge_labels[node1][node2], {"lb": -temporal_plan_network.edges[node1][node2], "ub": inf})
                elif temporal_plan_network.edges[node1][node2] == 0:
                    # Constraint duration = 0, adding as a lower bound on edge node2 -> node1
                    edge = Constraint(self.get_timepoint_by_id(node2), self.get_timepoint_by_id(node1), temporal_plan_network.edge_labels[node1][node2], {"lb": temporal_plan_network.edges[node1][node2], "ub": inf})
                self.add_constraint(edge)

        # If node is at the end of the network, manually adds an [0, inf] edge between the start time-point and it.
        start = self. get_timepoint_by_id(0)
        for timepoint in self.time_points:
            if not self.get_outgoing_edge_from_timepoint(timepoint):
                self.add_constraint(Constraint(start, timepoint, "Deadline for Timepoint {}".format(timepoint.id), {"lb": 0, "ub": inf}))
    
    def parse_from_json(self, json_file):
        """
        This function parses a JSON file and returns an instance of the temporal_network class.
        """
        if json_file[-5:] != ".json":
            json_file = json_file + ".json"

        # Opens the json and extracts the nodes and edge data
        with open(json_file) as f:
            data = json.load(f)
        self.name = data["name"]
        nodes, edges = data["timepoints"], data["constraints"]

        # Adds the nodes and edges
        for node in nodes:
            self.add_time_point(node["id"], node["label"])
        for edge in edges:
            source, sink = self.get_timepoint_by_id(edge["source"]), self.get_timepoint_by_id(edge["sink"])
            if edge["type"] == "pstc":
                raise AttributeError("Edge is of type 'probabilistic simple temporal constraint'. This constraint type is only valid for instances of Probabilistic Tempoal Network class.")
            to_add = Constraint(source, sink, edge["label"], {"lb": edge["duration_bound"]["lb"], "ub": edge["duration_bound"]["ub"]})
            self.add_constraint(to_add)

    def copy(self):
        """
        returns a copy of the temporal network.
        """
        tn = TemporalNetwork()
        tn.name = self.name
        tn.time_points = copy.deepcopy(self.time_points)
        tn.constraints = copy.deepcopy(self.constraints)
        return tn

    def add_time_point(self, time_point: TimePoint) -> None:
        """
        add a time-point (node) to the network.
        """
        for t in self.time_points:
            if t.id == time_point.id:
                raise ValueError("Time-point already exists in network with that ID. Try changing ID of new time-point so that it is unique.")
        self.time_points.append(time_point)
    
    def add_name(self, name: str) -> None:
        """
        adds a string name to the network.
        """
        self.name = name

    def add_constraint(self, constraint: Constraint) -> None:
        """
        add an edge (constraint) to the network. only constraints of type 'stc' are permitted in Temporal Network.
        if source and sink nodes not in the time_points set, it adds them
        """
        assert constraint.type == "stc", "Only time-points of the type 'stc' are permitted."
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
                existing.duration_bound["ub"] = constraint.ub
            elif constraint.lb > existing.lb:
                existing.duration_bound["lb"] = constraint.lb
        # If the source and sink time-points are the wrong way round in the new constraint versus existing
        elif existing.sink == constraint.source:
            if -constraint.lb < existing.ub:
                existing.duration_bound["ub"] = -constraint.lb
            if -constraint.ub > existing.lb:
                existing.duration_bound["lb"] = -constraint.ub
            
    def get_adjacency_matrix(self) -> dict[TimePoint, dict]:
        """
        gets adjacency matrix (dictionary) representation of temporal-network
        """
        adj = {}
        # Initialises adj matrix using edges explicit in self.constraints
        for constraint in self.constraints:
            # If source not in adj matrix yet, add it
            if constraint.source.id not in adj:
                adj[constraint.source.id] = {}
            # If sink not in adj matrix yet, add it
            if constraint.sink.id not in adj:
                adj[constraint.sink.id] = {}
            # If source[sink] not in adj matrix yet, add it.
            if constraint.sink.id not in adj[constraint.source.id]:
                adj[constraint.source.id][constraint.sink.id] = constraint.ub
            # If sink[source] not in adj matrix yet, add it
            if constraint.source.id not in adj[constraint.sink.id]:
                adj[constraint.sink.id][constraint.source.id] = -constraint.lb
        
        # Adds self edges to be equal to zero and initialises missing edges to be infinity
        for node1 in self.time_points:
            if node1.id not in adj:
                adj[node1.id] = {}
            for node2 in self.time_points:
                if node1 == node2:
                    adj[node1.id][node2.id] = 0
                elif node2.id not in adj[node1.id]:
                    adj[node1.id][node2.id] = inf
        return adj
    
    def get_bidirectional_network(self) -> dict[TimePoint, dict]:
        """
        Gets the bidirectional version of the temporal network, i.e. converts from l12 <= b2 - b1 <= u12 to b2 - b1 <= u12, b1 - b2 <= -l12
        As above but does not consider all pairs.
        """
        network = {}
        for constraint in self.constraints:
            # If source not in adj matrix yet, add it
            if constraint.source.id not in network:
                network[constraint.source.id] = {}
            # If sink not in adj matrix yet, add it
            if constraint.sink.id not in network:
                network[constraint.sink.id] = {}
            # If source[sink] not in adj matrix yet, add it.
            if constraint.sink.id not in network[constraint.source.id]:
                network[constraint.source.id][constraint.sink.id] = constraint.ub
            # If sink[source] not in adj matrix yet, add it
            if constraint.source.id not in network[constraint.sink.id]:
                network[constraint.sink.id][constraint.source.id] = -constraint.lb
        return network

    def check_consistency(self) -> bool:
        """
        use Floyd-Warshall to check consistency by detecting negative cycles.
        returns True if the network is consistent (i.e. no negative cycles).
        """
        adj = self.get_adjacency_matrix()

        # run Floyd-Warshall
        for k in adj:
            for i in adj:
                for j in adj:
                    adj[i][j] = min(adj[i][j], adj[i][k] + adj[k][j])
                    # check for negative cycles
                    if i==j and adj[i][j] < 0:
                        return False
        return True

    def floyd_warshall(self) -> tuple[dict[TimePoint, dict], bool]:
        """
        use Floyd-Warshall to put the graph in all-pairs shortest path form.
        """
        adj = self.get_adjacency_matrix()
        consistent = True
        # run Floyd-Warshall
        for k in adj:
            for i in adj:
                for j in adj:
                    adj[i][j] = min(adj[i][j], adj[i][k] + adj[k][j])
                    # check for negative cycles
                    if i==j and adj[i][j] < 0:
                        consistent = False

        # If consistent uses adjacency matrix to update edges.
        if consistent == True:
            for constraint in self.constraints:
                ub = adj[constraint.source.id][constraint.sink.id]
                lb = -adj[constraint.sink.id][constraint.source.id]
                constraint.duration_bound = {"lb": lb, "ub": ub}
        else:
            print("Not consistent: not updating")

    def find_shortest_path(self, source : int, sink : int) -> float:
        """
        find the shortest path using dijkstras search
        """
        # Needs updated
        network = self.get_bidirectional_network()
        distances = dict.fromkeys([i.id for i in self.time_points], float("inf"))
        distances[source.id] = 0
        queue = PriorityQueue()
        visited = set()
        queue.put((0, source))
        while not queue.empty():
            distance, node = queue.get()
            if node == sink: return distance
            if node in visited: continue
            visited.add(node)
            if node not in network: continue
            for neighbor in network[node]:
                if neighbor in visited: continue
                if distances[node] + network[node][neighbor] < distances[neighbor]:
                    distances[neighbor] = distances[node] + network[node][neighbor]
                    queue.put((distances[neighbor], neighbor))
        return float("inf")

    def make_minimal(self) -> dict[TimePoint]:
        """
        removes redundant edges from the network, assuming that the
        network is temporally consistent and already in all-pairs
        shortest path form.
        Reference:
        Nicola Muscettola, Paul Morris, and Ioannis Tsamardinos;
        "Reformulating Temporal Plans For Efficient Execution";
        In Principles of Knowledge Representation and Reasoning (1998).
        """
        check = self.floyd_warshall()
        adj, consistent = check[0], check[1]
        if consistent == False:
            raise AttributeError("Network is not consistent")    
        for k in adj:
            for i in adj:
                if i == k: continue
                if k not in adj[i]: continue
                for j in adj:
                    if i == j or j == k: continue
                    if j not in adj[k]: continue
                    if j not in adj[i]: continue
                    if adj[i][j] < adj[i][k] + adj[k][j]: continue
                    if adj[i][j] < 0 and adj[i][k] < 0:
                        del adj[i][j]
                    elif adj[i][j] >=0 and adj[k][j] >= 0:
                        del adj[i][j]
        return adj
    
    def get_outgoing_edge(self, constraint: Constraint) -> list[Constraint]:
        """
        given an edge (i, j), returns a list of outgoing edges (j, k)
        """
        return [jk for jk in self.constraints if jk.source == constraint.sink]
    
    def get_incoming_edge(self, constraint: Constraint) -> list[Constraint]:
        """
        given an edge (j, k), returns a list of incoming edges (i, j)
        """
        return [ij for ij in self.constraints if ij.sink == constraint.source]
    
    def get_outgoing_edge_from_timepoint(self, timepoint: TimePoint) -> list[Constraint]:
        """
        given a time-point i, returns a list of all outgoing edges (i, j)
        """
        return [ij for ij in self.constraints if ij.source == timepoint]

    def get_incoming_edge_from_timepoint(self, timepoint: TimePoint) -> list[Constraint]:
        """
        given a time-point j, returns a list of all incoming edges (i, j)
        """
        return [ij for ij in self.constraints if ij.sink == timepoint]

    def get_constraint_by_timepoint(self, source: TimePoint, sink: TimePoint) -> Constraint:
        """
        given two time-points, i and j, if a constraint exists between the two it returns the constraint, else raises exception 
        """
        for constraint in self.constraints:
            if constraint.source.id == source.id and constraint.sink.id == sink.id:
                return constraint
            elif constraint.sink.id == source.id and constraint.source.id == sink.id:
                return constraint
        return None

    def get_timepoint_by_id(self, id: int) -> TimePoint:
        """
        given an id, it returns the time-point if it exists in self.timepoints
        """
        found = None
        for time_point in self.time_points:
            if time_point.id == id:
                found = time_point
        return found
            
    def print_dot_graph(self):
        """
        print the graph in DOT format.
        """
        print("digraph G {")
        # declare nodes
        for time_point in self.time_points:
            print("\t" + str(time_point.id) + " [label=\"" + time_point.label + "\"];")
        # declare edges
        for constraint in self.constraints:
            print("\t{} -> {} [label=\"{}: [{}, {}]\"];".format(constraint.source.id, constraint.sink.id, constraint.label, constraint.lb, constraint.ub))
        print("}")
    
    def plot_dot_graph(self):
        """
        Plot the graph as a dot graph using graphviz
        """
        plot = Digraph()
        for timePoint in self.time_points:
            plot.node(name=str(timePoint.id), label=str(timePoint.id))
        
        for constraint in self.constraints:
            if constraint.ub == inf:
                plot.edge(str(constraint.source.id), str(constraint.sink.id), label="{}: [{:.3f}, inf]".format(constraint.label, constraint.lb))
            else:
                plot.edge(str(constraint.source.id), str(constraint.sink.id), label="{}: [{:.3f}, {:.3f}]".format(constraint.label, constraint.lb, constraint.ub))
        try:
            plot.render('junk/{}_plot.png'.format(self.name), view=True)
        except subprocess.CalledProcessError:
            print("Please close the PDF and rerun the script")

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

        with open(filename, 'w') as fp:
            json.dump(toDump, fp, indent=4, separators=(", ", ": "), cls=NpEncoder)