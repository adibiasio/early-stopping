from abc import abstractmethod
from typing import Callable

from strategies.AbstractStrategy import AbstractStrategy


class IterativeStrategy(AbstractStrategy):
    def __init__(
        self,
        sliding_window: int = 1,
        min_delta: float | int = 0,
        callbacks: list[Callable[[int, float, int, int], None]] = None,
    ):
        """
        Parameters:
        --------------
        sliding_window
            window size for averaging last n errors

        min_delta
            min difference in error to be classified as model relapse

        """
        super().__init__()

        if sliding_window and not isinstance(sliding_window, int):
            raise ValueError("Sliding window parameter must be an integer.")

        if min_delta and not isinstance(min_delta, (int, float)):
            raise ValueError("Minimum Delta parameter must be an integer or float.")

        self.sliding_window = sliding_window
        self.min_delta = min_delta
        self.callbacks = callbacks

    # TODO: ensure that for patience of 1 million and 2 million there are the same ranks
    # TODO: only update patience value when new best iteration has been found
    def _run(self, curve: list[float]):
        """
        Parameters:
        --------------
        curve

        Return:
        --------------
        best iteration (zero indexed)

        total iterations (zero indexed)
        """
        best_iter = 0
        best_error = None
        sliding_sum = 0
        iter_wo_improvement = 0
        patience = self.patience(1)

        self.runCallbacks("before_simulation", strategy=self)

        for iter, error in enumerate(curve):
            self.runCallbacks(
                "before_iter",
                strategy=self,
                iter=iter,
                metric=error,
                iter_wo_improvement=iter_wo_improvement,
                patience=patience,
            )

            if self.sliding_window > 1:
                n = min(iter + 1, self.sliding_window)
                sliding_sum += error
                sliding_sum -= curve[iter - n] if iter - n >= 0 else 0
                error = sliding_sum / n

            if best_error is None:
                best_error = error

            elif error >= best_error + self.min_delta:
                iter_wo_improvement += 1
                if iter_wo_improvement >= patience:
                    # after iteration callbacks
                    self.runCallbacks(
                        "after_iter",
                        strategy=self,
                        iter=iter,
                        metric=error,
                        iter_wo_improvement=iter_wo_improvement,
                        patience=patience,
                    )
                    break

            else:
                best_iter = iter
                best_error = error
                iter_wo_improvement = 0
                patience = self.patience(iter + 1)

            # after iteration callbacks
            self.runCallbacks(
                "after_iter",
                strategy=self,
                iter=iter,
                metric=error,
                iter_wo_improvement=iter_wo_improvement,
                patience=patience,
            )

        self.runCallbacks("after_simulation", strategy=self)
        return best_iter, iter

    def runCallbacks(self, method: str, **kwargs):
        if self.callbacks:
            for callback in self.callbacks:
                func = getattr(callback, method)
                if func(**kwargs):
                    break

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

    @property
    def patience(self) -> Callable[[int], int]:
        return lambda x: 0

    @classmethod
    def kwargs(cls) -> dict[str, str]:
        kwargs = super().kwargs()
        kwargs.update({
            "sliding_window": "sw",
            "min_delta": "md",
        })
        return kwargs
