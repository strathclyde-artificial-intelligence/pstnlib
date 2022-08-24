from ast import Dict
import copy
from pstnlib.temporal_networks.timepoint import TimePoint
from scipy import stats
inf = 1000000000

class Constraint:
    """
    represents a temporal network constraint (edge in the network)
    """
    def __init__(self, source: TimePoint, sink: TimePoint, label: str, duration_bound: dict[str, str], distribution: dict[str, str] = None):
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
        return Constraint(self.source, self.sink, self.label[:], self.duration_bound)

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
    def __init__(self, source: TimePoint, sink: TimePoint, label: str, distribution: dict) -> None:
        super().__init__(source, sink, label, {"lb": 0, "ub": inf})
        self.type = "pstc"
        assert list(distribution.keys()) == ["mean", "sd"],  "Distribution should be in the form {'mean': float, 'sd': float}"
        self.distribution = distribution
        self.approximation = {"points": [], "evaluations": []}

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

    def set_distribution(self, distribution: dict[str, str]) -> None:
        """
        used to change the distribution of the edge if probabilistic
        """
        assert distribution.keys() == ["mean", "sd"],  "Distribution should be in the form {'mean': float, 'sd': float}"
        self.distribuion = distribution
    
    def evaluate_probability(self, l, u):
        distribution = stats.normal(self.mean, self.sd)
        return distribution.cdf(u) - distribution.cdf(l)

    @property
    def mean(self):
        return self.distribution["mean"]

    @property
    def sd(self):
        return self.distribution["sd"]

        self, source: TimePoint, sink: TimePoint, label: str, duration_bound: dict[str, str], distribution: dict[str, str] = None