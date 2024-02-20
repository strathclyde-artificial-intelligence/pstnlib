class TimePoint:
    """
    Class representing a time point (vertex in the temporal network)
    ------------------------
    Params:
        id:             int
                            unique identifier of node in the temporal network.
        label:          str
                            string used to label the node.
        controllable:   bool
                            True if time point is controllable (we can choose when to schedule it), else False.

    """
    def __init__(self, id: int, label: str, controllable: bool = True) -> None:
        self.id = id
        self.label = label
        self.controllable = controllable
    
    def copy(self):
        """
        returns a copy of the time-point
        """
        return TimePoint(self.id, self.label[:])

    def __str__(self) -> str:
        """
        prints string representation of time-point
        """
        return "Time-point {}".format(self.id)
    
    def to_json(self) -> str:
        """
        prints the time-point as a dictionary for use with json
        """
        return {"id": self.id, "label": self.label}