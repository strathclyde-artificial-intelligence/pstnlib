from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
from pstnlib.optimisation.pstn_optimisation_class import PstnOptimisation
from pstnlib.optimisation.paris import paris
from pstnlib.optimisation.solution import Solution
import sys
from time import time

if __name__ == "__main__":
    """
    This script solves a case using column generation and LP and saves results
    """
    # command line arguments
    if len(sys.argv) != 3:
        raise ValueError("Script should take two arguments:\n\t 1. The path to the network to be solved.\n\t 2. The path to the directory to store the result.")

    cstn = CorrelatedTemporalNetwork()
    cstn.parse_from_json(sys.argv[1])

    # # # # Optimises using column generation
    op = PstnOptimisation(cstn, verbose=True)
    op.optimise()
    convex = op.solutions[-1]
    convex.to_json(sys.argv[2])

    # # # # Optimises using PARIS Algorithm.
    start = time()
    m = paris(cstn)
    lp = Solution(cstn, m, time() - start)
    lp.to_json(sys.argv[2])