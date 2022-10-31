from random_generation import generate_random_cstn
import numpy as np

def main():
    domain = "temporal-planning-domains/drones/domain.pddl"
    problem_dir = "temporal-planning-domains/drones/instances/"
    plan_dir = "temporal-planning-domains/drones/plans/"
    output_dir = "temporal-planning-domains/drones/networks/"
    no_cstns = 10
    problems = ["instance-29.pddl", "instance-31.pddl", "instance-32.pddl", "instance-33.pddl", "instance-35.pddl", "instance-36.pddl", "instance-40.pddl", "instance-41.pddl",
                "instance-42.pddl", "instance-51.pddl", "instance-54.pddl", "instance-56.pddl", "instance-61.pddl", "instance-66.pddl", "instance-67.pddl", "instance-72.pddl",
                "instance-73.pddl", "instance-77.pddl", "instance-79.pddl", "instance-105.pddl", "instance-114.pddl", "instance-117.pddl", "instance-118.pddl", "instance-136.pddl",]
    
    tightness_factors = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
    corr_sizes = [2, 3, 4]
    for problem in problems:
        tokens = problem.split(".")
        plan = "drones_{}_plan.pddl".format(tokens[0])
        for i in range(1, no_cstns + 1):
            for size in corr_sizes:
                for factor in tightness_factors:
                    network = generate_random_cstn(domain, problem_dir + problem, plan_dir + plan, corr_size=size, tightness_factor=factor)
                    network.name = "drones_{}_network_{}_deadline_{}_corrsize_{}".format(problem.split(".")[0], i, ("").join(str(round(factor, 1)).split(".")), size)
                    network.save_as_json(output_dir + network.name + ".json")

if __name__ == "__main__":
    main()