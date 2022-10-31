from audioop import reverse
from graphlib import TopologicalSorter
from inspect import formatargvalues
import random
import this
from xml.dom.minidom import Attr
import numpy as np
from pyparsing import delimited_list
from scipy import stats
from otpl.pddl.parser import Parser
import json
from pstnlib.temporal_networks.constraint import Constraint
from pstnlib.temporal_networks.timepoint import TimePoint
from pstnlib.temporal_networks.correlation import Correlation
from pstnlib.temporal_networks.temporal_network import TemporalNetwork
from pstnlib.temporal_networks.probabilistic_temporal_network import ProbabilisticTemporalNetwork
from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
from otpl.plans.temporal_plan import PlanTemporalNetwork, Happening, HappeningType
from otpl.temporal_networks.simple_temporal_network import SimpleTemporalNetwork

inf = 1000000000

def generate_problem(no_drones, no_locations, no_depots, no_medicines, file):
    drones = []
    locations = []
    medicines = []
    distances = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    distance_probs = [0.05, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]
    drone_types = {"small": {"load-capacity": 10, "battery-capacity": 50, "battery-rate": 1, "recharge-rate": 10},
                    "medium": {"load-capacity": 20, "battery-capacity": 100, "battery-rate": 1, "recharge-rate": 5},
                     "large": {"load-capacity": 50, "battery-capacity": 150, "battery-rate": 1, "recharge-rate": 4}}
    medicine_types = {"penicillin": {"weight": 2, "probability": 0.25, "expiration time": 400},
                        "insulin": {"weight": 1, "probability": 0.15, "expiration time": 180},
                            "defibrillator": {"weight": 20, "probability": 0.05, "expiration time": 100},
                                "blood": {"weight": 10, "probability": 0.15, "expiration time": 120},
                                    "organ": {"weight": 20, "probability": 0.1, "expiration time": 100},
                                        "vaccine": {"weight": 2, "probability": 0.1, "expiration time": 150},
                                            "atorvastatin": {"weight": 2, "probability": 0.05, "expiration time": 200},
                                                "levothyroxine": {"weight": 3, "probability": 0.05, "expiration time": 300},
                                                    "metformin": {"weight": 5, "probability": 0.1, "expiration time": 500}}

    with open(file, "w") as f:
        name = file.split("/")[-1]
        f.write("(define (problem {})\n".format(name[:-5]))
        f.write("(:domain drone-delivery)\n")

        f.write("(:objects\n ")

        for i in range(no_drones):
            drone = "d"+str(i)
            drones.append(drone)
            f.write(drone+" ")
        f.write("- drone\n ")

        for i in range(no_locations):
            location = "l"+str(i)
            locations.append(location)
            f.write(location+" ")
        f.write("- location\n ")

        for i in range(no_medicines):
            medicine = "m"+str(i)
            medicines.append(medicine)
            f.write(medicine+" ")
        f.write("- medicine\n ")

        f.write(")\n")
        # end of objects

        f.write("(:init\n")

        f.write("\n\t;; depots\n")
        depots = random.sample(locations, no_depots)
        for depot in depots:
            f.write("\t(is-depot " + depot + ")\n")

        f.write("\n\t;; drones\n")
        # Randomly splits drones into the three categories.
        drones_copy = drones[:]
        random.shuffle(drones_copy)
        small_d = drones_copy[:random.randint(0, len(drones_copy))]
        medium_d = drones_copy[len(small_d):random.randint(len(small_d), len(drones_copy))]
        large_d = drones_copy[len(small_d) + len(medium_d):len(drones_copy)]
        # Assigns drone locations and parameters based on drone classification.
        for drone in drones:
            location = random.choice(depots)
            f.write("\t(located-at " + drone + " " + depot + ")\n")
            f.write("\t(noloading " + drone + ")\n")
            f.write("\t(nocharging " + drone + ")\n")
            if drone in small_d:
                f.write("\t(= (load-capacity " + drone + ") {})\n".format(drone_types["small"]["load-capacity"]))
                f.write("\t(= (battery-capacity " + drone + ") {})\n".format(drone_types["small"]["battery-capacity"]))
                f.write("\t(= (battery-level " + drone + ") {})\n".format(drone_types["small"]["battery-capacity"]))
                f.write("\t(= (battery-rate " + drone + ") {})\n".format(drone_types["small"]["battery-rate"]))
                f.write("\t(= (recharge-rate " + drone + ") {})\n".format(drone_types["small"]["recharge-rate"]))
            elif drone in medium_d:
                f.write("\t(= (load-capacity " + drone + ") {})\n".format(drone_types["medium"]["load-capacity"]))
                f.write("\t(= (battery-capacity " + drone + ") {})\n".format(drone_types["medium"]["battery-capacity"]))
                f.write("\t(= (battery-level " + drone + ") {})\n".format(drone_types["medium"]["battery-capacity"]))
                f.write("\t(= (battery-rate " + drone + ") {})\n".format(drone_types["medium"]["battery-rate"]))
                f.write("\t(= (recharge-rate " + drone + ") {})\n".format(drone_types["medium"]["recharge-rate"]))
            else:
                f.write("\t(= (load-capacity " + drone + ") {})\n".format(drone_types["large"]["load-capacity"]))
                f.write("\t(= (battery-capacity " + drone + ") {})\n".format(drone_types["large"]["battery-capacity"]))
                f.write("\t(= (battery-level " + drone + ") {})\n".format(drone_types["large"]["battery-capacity"]))
                f.write("\t(= (battery-rate " + drone + ") {})\n".format(drone_types["large"]["battery-rate"]))
                f.write("\t(= (recharge-rate " + drone + ") {})\n".format(drone_types["large"]["recharge-rate"]))

        f.write("\n\t;; medicines\n")
        start_locations = {}
        for medicine in medicines:
            location = random.choice(locations)
            start_locations[medicine] = location
            types = [k for k in medicine_types]
            probs = [medicine_types[k]["probability"] for k in medicine_types]
            medicine_type = np.random.choice(types, p = probs)
            f.write("\t(located-at " + medicine + " " + location + ")\n")
            f.write("\t(noexpired " + medicine + ")\n")
            if medicine_types[medicine_type]["expiration time"] != None:
                expiration_time = medicine_types[medicine_type]["expiration time"]
                f.write("\t(at {}(not (noexpired ".format(expiration_time) + medicine + ")))\n")
            f.write("\t(= (weight " + medicine + ") {})\n".format(medicine_types[medicine_type]["weight"]))

        f.write("\n\t;; locations\n")
        topology = {}
        for l1 in locations:
            topology[l1] = {}
            for l2 in locations:
                if l1 != l2:
                    if l2 in topology:
                        if l1 in topology[l2]:
                            topology[l1][l2] = topology[l2][l1]
                    else:
                        distance = np.random.choice(distances, p = distance_probs)
                        is_connected = random.uniform(0, 1)
                        if is_connected < 0.5:
                            topology[l1][l2] = distance
                        
        for l1 in topology:
            for l2 in topology[l1]:
                f.write("\t(connected " + l1 + " " + l2 + ")\n")
                f.write("\t(= (travel-time " + l1 + " " + l2 + ") {})\n".format(topology[l1][l2]))
        f.write(")\n")
        # end of initial state

        f.write("(:goal (and\n")
        delivery_points = [l for l in locations if l not in depots]
        for medicine in medicines:
            delivery_point = random.choice([dp for dp in delivery_points if dp != start_locations[medicine]])
            f.write("\t(delivered " + medicine + " " + delivery_point + ")\n")
        f.write(")))\n")
        f.close()

def generate_random_uncertainties(domain_file: str, problem_file: str) -> dict:
    """
    randomly generates a dictionary of uncertainties for action and til durations.
    """
    # Parses domain and problem file
    pddl_parser = Parser()
    pddl_parser.parse_file(domain_file)
    pddl_parser.parse_file(problem_file)
    problem, domain = pddl_parser.problem, pddl_parser.domain
    
    # Extracts list of tils and action names
    # tils = problem.timed_initial_literals
    actions = [a for a in domain.operators]

    # Randomly selects tils to be made probabilistic
    #no_to_be_randomised = random.randint(min(1, len(tils)), len(tils))
    # no_to_be_randomised = len(tils)
    # random_tils = random.sample(tils, no_to_be_randomised)

    # Randomly selects actions to be made probabilistic
    #no_to_be_randomised = random.randint(min(1, len(actions)), len(actions))
    no_to_be_randomised = len(actions)
    random_actions = random.sample(actions, no_to_be_randomised)

    # Randomly generates uncertainties. Uncertainties are added as a fraction of the duration  given in the plan.
    # If action takes 10 in plan and mean_fraction is 0.8, then the mean is 8.
    uncertainties = {"actions": [], "tils": []}
    for action in random_actions:
        if "move" in action:
            to_add = {"name": action, "mean_fraction": abs(np.random.normal(1, 0.2)), "sd_fraction": abs(np.random.normal(0.15, 0.05))}
            uncertainties["actions"].append(to_add)
    
    # for til in random_tils:
    #     to_add = {"name": til, "mean_fraction": abs(np.random.normal(1, 0.2)), "sd_fraction": abs(np.random.normal(0.15, 0.05))}
    return uncertainties

def save_random_uncertainties(domain_file: str, problem_file: str, output_file: str) -> None:
    """
    wrapper for generate_random_uncertainties to save as json
    """
    if output_file[-5:] != ".json":
            output_file = output_file + ".json"
    uncertainties = generate_random_uncertainties(domain_file, problem_file)
    with open(output_file, "w") as f:
        data = json.dump(uncertainties, f, indent=4, separators=(", ", ": "))

def sample_probabilistic_constraints(network: ProbabilisticTemporalNetwork, n_correlations: int, size: int) -> list[Constraint]:
    """
    Randomly generates "n_correlations" samples of the probabilistic constraints in the network. Size is the number of constraints in each sample.
    """
    if len(network.get_probabilistic_constraints()) < n_correlations * size:
        raise ValueError("Insuffucient probabilistic constraints for required size.")
    sample = random.sample(network.get_probabilistic_constraints(), n_correlations * size)
    random.shuffle(sample)
    partitions = [sample[i::n_correlations] for i in range(n_correlations)]
    return partitions

def generate_random_constraints(network: TemporalNetwork, deadline: float, number: int) -> list[Constraint]:
    """
    Randomly generates a set of constraints of size n to add to the temporal network. 
    """
    cs = [c for c in network.constraints if "Ordering" in c.label or "Interference" in c.label]
    sample = random.sample(cs, number)
    consistent = False
    while consistent == False:
        for constraint in sample:
            ub = random.uniform(0, deadline)
            lb = random.uniform(0, ub)
            constraint.duration_bound = {"lb": lb, "ub": ub}
        if network.check_consistency() == True:
            consistent = True
    return network

def generate_random_cstn(domain_f: str, problem_f: str, plan_f: str, corr_size = 2, tightness_factor = 1.0):
    """
    Generates a random simple temporal network in all pairs shortest path form from a given pddl domain, problem and plan.
    """
    instance = plan_f.split("/")[-1]
    instance = instance.split(".")[0]
    pddl_parser = Parser()
    pddl_parser.parse_file(domain_f)
    pddl_parser.parse_file(problem_f)

    # Gets the tils representing expiration of a medical supply.
    tils = [t.__repr__() for t in pddl_parser.problem.timed_initial_literals if "(not (noexpired" in t.__repr__()]

    # parses plan and outputs simple temporal network.
    plan = PlanTemporalNetwork(pddl_parser.domain, pddl_parser.problem)
    plan.read_from_file(plan_f)
    plan.temporal_network.make_minimal()

    # Gets list of medicines to be delivered and the locations they should be delivered.
    medicines = [k for k in pddl_parser.problem.objects_type_map if pddl_parser.problem.objects_type_map[k] == "medicine"]
    locations = {}
    for medicine in medicines:
        for goal in plan.problem.goal.goals:
            if "delivered {}".format(medicine) in goal.__str__():
                locations[medicine] = goal.__str__().split(" ")[-1][:-1]

    # Collects the edges representing the expiry of a medicine.
    til_edges = {}
    for medicine in medicines:
        for til in tils:
            for sink in plan.temporal_network.edge_labels[0]:
                if plan.temporal_network.edge_labels[0][sink][5:] in til and "not " in plan.temporal_network.edge_labels[0][sink] and medicine in til:
                    til_edges[medicine] = (0, sink)

    # Gets duration between when the medicine was picked up and dropped off.
    delivery_duration = {}
    for medicine in medicines:
        delivery_duration[medicine] = {}
        for happening in plan.time_sorted_happenings:
            # Gets time it was delivered according to plan.
            if happening.type == HappeningType.ACTION_END:
                formula = plan.grounding.get_action_from_id(happening.action_id).print_pddl()
                if "complete-delivery" in formula and medicine in formula:
                    if "end" in delivery_duration[medicine]:
                        delivery_duration[medicine]["time"] = min(happening.time, delivery_duration[medicine]["end"])
                    else:
                        delivery_duration[medicine]["time"] = happening.time
            #If its dropped off earlier it updates the end of the deadline with the time it was dropped at the goal.
                if "drop-off" in formula and medicine in formula and locations[medicine] in formula:
                    if "end" in delivery_duration[medicine]:
                        delivery_duration[medicine]["time"] = min(happening.time, delivery_duration[medicine]["end"])
                    else:
                        delivery_duration[medicine]["time"] = happening.time
    
    # Updates expiry TIL edges so that they are tight.
    for medicine in medicines:
        til = til_edges[medicine]
        deadline = delivery_duration[medicine]["time"]
        plan.temporal_network.edges[til[0]][til[1]] = deadline * tightness_factor
        plan.temporal_network.edges[til[1]][til[0]] = -deadline * tightness_factor

    # parses simple temporal network and makes instance of temporal network
    network = CorrelatedTemporalNetwork()
    network.parse_from_temporal_plan_network(plan.temporal_network)

    # # Generates and adds random uncertainties.
    uncertainties = generate_random_uncertainties(domain_f, problem_f)
    network.parse_uncertainties_from_dict(uncertainties)

    # If both start and end timepoints are uncontrollable it decomposes the constraint containing the beginning timepoint.
    for constraint in network.get_requirement_constraints():
        incoming = network.get_incoming_probabilistic(constraint)
        if incoming["start"] != None and incoming["end"] != None:
            # Make new controllable timepoint
            index = max([t.id for t in network.time_points]) + 1
            intermediate_timepoint_start = TimePoint(index, "Intermediate timepoint {}".format(index))
            network.add_time_point(intermediate_timepoint_start)
            # Udates probabilistic sink to new timepoint
            incoming["start"].sink = intermediate_timepoint_start
            # Makes constraint between probabilistic sink and uncontrollable start
            network.add_constraint(Constraint(intermediate_timepoint_start, constraint.source, "Intermediate constraint added for {}".format(constraint.get_description()), {"lb": 0, "ub": inf}))

    # Gets list of drones
    drones = [k for k in pddl_parser.problem.objects_type_map if pddl_parser.problem.objects_type_map[k] == "drone"]
    # Gets list of actions associated with drone moving 
    for drone in drones:
        action_edges = [c for c in network.get_probabilistic_constraints() if "move" in c.label and drone in c.label]
        means = [c.mean for c in action_edges]
        sorted_action_edges = [c for _, c in sorted(zip(means, action_edges), key=lambda pair: pair[0], reverse=True)]
        try:
            correlated_constraints = sorted_action_edges[:corr_size]
            corr = Correlation(correlated_constraints)
            corr.add_random_correlation()
            network.add_correlation(corr)
        except IndexError:
            correlated_constraints = sorted_action_edges
            corr = Correlation(correlated_constraints)
            corr.add_random_correlation()
            network.add_correlation(corr)
        except ValueError:
            continue
    return network