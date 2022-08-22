class TimePoint:
    """
    represents a time point (vertex in the temporal network)
    """
    def __init__(self, id: int, label: str, controllable: bool = True) -> None:
        # controllable: True if time-point can be scheduled, 
        #               False if it cannot (follows an uncertain duration)
        #               Always True for STN
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