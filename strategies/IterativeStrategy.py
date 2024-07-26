from abc import abstractmethod
from typing import List, Callable

from strategies.AbstractStrategy import AbstractStrategy


class IterativeStrategy(AbstractStrategy):
    def __init__(self, patience: Callable[[int], int] = lambda x : 0, sliding_window: int = 1, min_delta: float | int = 0):
        """
        Parameters:
        --------------
        patience
            patience function

        sliding
            window size for averaging last n errors

        min_delta
            min difference in error to be classified as model relapse

        """
        super().__init__()

        if sliding_window and not isinstance(sliding_window, int):
            raise ValueError("Sliding window parameter must be an integer.")

        if min_delta and not isinstance(min_delta, (int, float)):
            raise ValueError("Minimum Delta parameter must be an integer or float.")

        if patience and not callable(patience):
            raise ValueError("Patience function must be of format p(int) -> int")

        self.patience = patience
        self.sliding_window = sliding_window
        self.min_delta = min_delta


    # TODO: ensure that for patience of 1 million and 2 million there are the same ranks
    def _run(self, curve: List[float]):
        """
        Parameters:
        --------------
        curve
        
        Return:
        --------------
        best iteration (zero indexed)

        total iterations (zero indexed)
        """
        counter = 0
        best_iter = 0
        best_error = None
        sliding_sum = 0
        for iter, error in enumerate(curve):
            if self.sliding_window > 1:
                n = min(iter + 1, self.sliding_window)
                sliding_sum += error
                sliding_sum -= curve[iter - n] if iter - n >= 0 else 0
                error = sliding_sum / n
            if best_error is None:
                best_error = error
            elif error >= best_error + self.min_delta:
                counter += 1
                if counter >= self.patience(iter + 1):
                    break
            else:
                best_iter = iter
                best_error = error
                counter = 0

        return best_iter, iter


    @property
    def name(self):
        base = self._base_name()

        sliding = self.sliding_window != 1
        min_delta = self.min_delta != 0

        if base == "min_delta":
            min_delta = False
        elif base == "sliding_window":
            sliding = False

        if sliding:
            base = "sliding_window_" + base

        if min_delta:
            base += "_with_min_delta"

        return base


    @abstractmethod
    def _base_name(self):
        pass


    @classmethod
    def user_params(cls) -> "set[str]":
        return super().user_params().union({ "sliding_window", "min_delta" })
