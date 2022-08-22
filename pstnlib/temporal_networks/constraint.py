import copy
from pstnlib.temporal_networks.timepoint import TimePoint

class Constraint:
    """
    represents a temporal network constraint (edge in the network)
    """
    def __init__(self, source: TimePoint, sink: TimePoint, label: str, type: str, duration_bound: dict[str, str], distribution: dict[str, str] = None):
        self.source = source
        self.sink = sink
        self.label = label
        assert type in ("stc, pstc"), "Invalid Constraint type, type must be 'stc' for simple temporal constraint, or 'pstc' for probabilistic simple temporal constraint"
        self.type = type
        assert list(duration_bound.keys()) == ["lb", "ub"],  "Duration_bound should be in the form {'lb: float, 'ub': float}"
        self.duration_bound = duration_bound
        if distribution != None:
            assert list(distribution.keys()) == ["mean", "sd"],  "Distribution should be in the form {'mean': float, 'sd': float}"
        self.distribution = distribution
        
    def get_description(self) -> str:
        """
        returns a string of the from c(source.id, sink.id)
        """
        return "c({},{})".format((self.source.id), (self.sink.id))
    
    def copy_constraint(self):
        """
        returns a copy of the constraint
        """
        return Constraint(self.label[:], self.source.copy(), self.sink.copy(), self.type[:], copy.deepcopy(self.duration_bound), distribution = copy.deepcopy(self.distribution))
    
    def set_type(self, type: str) -> None:
        """
        used to change the type of the constraint from stc to pstc or vis versa
        """
        assert type in ("stc, pstc"), "Invalid Constraint type, type must be 'stc' for simple temporal constraint, 'stcu' for simple temporal constraint with uncertainty or 'pstc' for probabilistic simple temporal constraint"
        self.type = type
    
    def set_distribution(self, distribution: dict[str, str]) -> None:
        """
        used to change the distribution of the edge if probabilistic
        """
        if self.type == "pstc":
            assert distribution.keys() == ["mean", "sd"],  "Distribution should be in the form {'mean': float, 'sd': float}"
            self.distribuion = distribution
        else:
            raise AttributeError("Constraint is not of type pstc. Please use constraint.set_type('pstc') first if you wish to change it")
    
    def __str__(self) -> None:
        """
        used to print the constraint in a user-friendly way
        """
        if self.type == "stc":
            return self.get_description() + ": " + "[{}, {}] ".format(self.duration_bound["lb"], self.duration_bound["ub"])
        elif self.type == "pstc":
            return self.get_description() + ": " + "N({}, {})".format(self.distribution["mean"], self.distribution["variance"])

    def to_json(self) -> dict:
        """
        returns the constraint as a dictionary for use with json
        """
        to_return = {"source": self.source.id, "sink": self.sink.id, "label": self.label, "type": self.type, "duration_bound": {"lb": self.lb, "ub": self.ub}}
        if self.type == "pstc":
            to_return["distribution"] = {"mean": self.mean, "sd": self.sd}
        return to_return

    @property
    def mean(self):
        if self.distribution != None:    
            return self.distribution["mean"]
        else:
            raise ValueError("Constraint is not probabilistic and so has no mean")

    @property
    def sd(self):
        if self.distribution != None:    
            return self.distribution["sd"]
        else:
            raise ValueError("Constraint is not probabilistic and so has no standard deviation")
    
    @property
    def lb(self):
        return self.duration_bound["lb"]

    @property
    def ub(self):
        return self.duration_bound["ub"]