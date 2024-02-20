import copy
from pstnlib.temporal_networks.timepoint import TimePoint
from scipy import stats
import numpy as np
inf = 1000000000

class Constraint:
    """
    Represents a temporal network constraint (edge in the network).
    -----------------------
    Params:
        source:         pstnlib.temporal_networks.timepoint.TimePoint
                            source node in edge
        sink:           pstnlib.temporal_networks.timepoint.TimePoint
                            sink node in edge
        label:          str
                            label describing edge.
        type:           str
                            string type of edge. should be stc for simple temporal constraint or pstc for probabilistic simple temporal constrai.
        duration_bound: dict
                            duration_bound for constraint in the form {'lb: float, 'ub': float}           
    """
    def __init__(self, source: TimePoint, sink: TimePoint, label: str, duration_bound: dict):
        self.source = source
        self.sink = sink
        self.label = label
        self.type = "stc"
        assert list(duration_bound.keys()) == ["lb", "ub"],  "Duration_bound should be in the form {'lb: float, 'ub': float}"
        self.duration_bound = duration_bound
        
    def get_description(self) -> str:
        """
        returns a string of the from c(source.id, sink.id)
        """
        return "c({},{})".format((self.source.id), (self.sink.id))
    
    def copy_constraint(self):
        """
        returns a copy of the constraint
        """
        return Constraint(self.source, self.sink, self.label[:], copy.deepcopy(self.duration_bound))

    def copy_as_probabilistic(self, distribution):
        """
        returns a copy as an instance of ProbabilisticConstraint class with defined distribution.
        """
        return ProbabilisticConstraint(self.source, self.sink, self.label[:], distribution)

    def __str__(self) -> None:
        """
        used to print the constraint in a user-friendly way
        """
        return self.get_description() + ": " + "[{}, {}] ".format(self.lb, self.ub)

    def to_json(self) -> dict:
        """
        returns the constraint as a dictionary for use with json
        """
        return {"source": self.source.id, "sink": self.sink.id, "label": self.label, "type": self.type, "duration_bound": {"lb": self.lb, "ub": self.ub}}

    @property
    def lb(self):
        return self.duration_bound["lb"]

    @property
    def ub(self):
        return self.duration_bound["ub"]

class ProbabilisticConstraint(Constraint):
    """
    Represents a probabilistic constraint (edge in the network).
    -----------------------
    Params:
        source:         pstnlib.temporal_networks.timepoint.TimePoint
                            source node in edge
        sink:           pstnlib.temporal_networks.timepoint.TimePoint
                            sink node in edge
        label:          str
                            label describing edge.
        distribution:   dict
                            dictionary of mean and standard deviation.
        approximation:  dict
                            dictionary of approximation vectors and their probabilities.      
    """
    def __init__(self, source: TimePoint, sink: TimePoint, label: str, distribution: dict) -> None:
        super().__init__(source, sink, label, {"lb": 0, "ub": inf})
        self.type = "pstc"
        assert list(distribution.keys()) == ["mean", "sd"],  "Distribution should be in the form {'mean': float, 'sd': float}"
        self.distribution = distribution
        self.approximation = None

    def __str__(self) -> None:
        """
        used to print the constraint in a user-friendly way
        """
        return self.get_description() + ": " + "N({}, {})".format(self.mean, self.sd)

    def to_json(self) -> dict:
        """
        returns the constraint as a dictionary for use with json
        """
        return {"source": self.source.id, "sink": self.sink.id, "label": self.label, "type": self.type, "distribution": {"mean": self.mean, "sd": self.sd}}

    def set_distribution(self, distribution: dict) -> None:
        """
        used to change the distribution of the edge if probabilistic
        """
        assert distribution.keys() == ["mean", "sd"],  "Distribution should be in the form {'mean': float, 'sd': float}"
        self.distribuion = distribution
    
    def evaluate_probability(self, l, u):
        """
        Calculates the probability F(u) - F(l), where F(z) is the cumulative probability function of the probabilistic constraint.
        """
        z = np.array([u, -l])
        mean = np.array([self.mean, -self.mean])
        var = self.sd**2
        cov = np.array([[var, -var], [-var, var]])
        return stats.multivariate_normal(mean, cov, allow_singular=True).cdf(z)

    def evaluate_gradient(self, l, u):
        '''
        Calculates the gradient of the function F(u) - F(l).
        '''
        distribution = stats.norm(self.mean, self.sd)
        return (-distribution.pdf(l), distribution.pdf(u))

    def add_approximation_point(self, l, u, phi):
        """
        If approximation point does not already exist in the approximation
        we add it.
        """
        for point in self.approximation["points"]:
            if point[0] == l and point[1] == u:
                return False
        self.approximation["points"].append((l, u))
        self.approximation["evaluation"].append(phi)
        return True
            
    @property
    def mean(self):
        return self.distribution["mean"]

    @property
    def sd(self):
        return self.distribution["sd"]