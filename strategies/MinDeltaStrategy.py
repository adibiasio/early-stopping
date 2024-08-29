from .IterativeStrategy import IterativeStrategy


class MinDeltaStrategy(IterativeStrategy):
    _name = "min_delta"
    _short_name = "MD"

    def __init__(self, min_delta: float = 0, sliding_window: int = 1):
        super().__init__(min_delta=min_delta, sliding_window=sliding_window)
