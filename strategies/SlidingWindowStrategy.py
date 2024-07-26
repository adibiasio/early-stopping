from strategies.IterativeStrategy import IterativeStrategy


class SlidingWindowStrategy(IterativeStrategy):
    def __init__(self, sliding_window: int = 1, min_delta: float = 0):
        super().__init__(sliding_window=sliding_window, min_delta=min_delta)


    def _base_name(self):
        return "sliding_window"


    @classmethod
    def user_params(cls) -> "set[str]":
        return { "min_delta", "sliding_window" }
