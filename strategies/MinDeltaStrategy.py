from strategies.IterativeStrategy import IterativeStrategy


class MinDeltaStrategy(IterativeStrategy):
    def __init__(self, min_delta: float = 0, sliding_window: int = 1):
        super().__init__(min_delta=min_delta, sliding_window=sliding_window)


    def _base_name(self):
        return "min_delta"


    @classmethod
    def user_params(cls) -> "set[str]":
        return { "min_delta", "sliding_window" }
