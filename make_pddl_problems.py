from random_generation import generate_problem

def main():
    i = 1
    for medicines in [1, 2, 4, 8]:
        for drones in [1, 2, 3, 4]:
            for j in range(10):
                generate_problem(drones, 10, 2, medicines, "temporal-planning-domains/drones/instances/instance-{}.pddl".format(i))
                i += 1

if __name__ == "__main__":
    main()

