from pstnlib.temporal_networks.probabilistic_temporal_network import ProbabilisticTemporalNetwork
from pstnlib.temporal_networks.correlation import Correlation
from pstnlib.temporal_networks.constraint import Constraint

class CorrelatedTemporalNetwork(ProbabilisticTemporalNetwork):
    """
    represents a correlated probabilistic temporal network.
    """
    def __init__(self) -> None:
        super().__init__()
        self.correlations = []
    
    def add_correlation(self, constraints: list[Constraint]) -> None:
        self.correlations.append(Correlation(constraints))