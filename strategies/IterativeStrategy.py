from __future__ import annotations

from abc import abstractmethod
from typing import Callable

from .AbstractStrategy import AbstractStrategy


class IterativeStrategy(AbstractStrategy):
    def __init__(
        self,
        n_iter: int = 0,
        sliding_window: int = 1,
        min_delta: float | int = 0,
        callbacks: list | None = None,
    ):
        """
        Parameters:
        --------------
        n_iter
            maximum number of iterations to train for. Unlimited if 0.

        sliding_window
            window size for averaging last n errors

        min_delta
            min difference in error to be classified as model relapse

        """
        super().__init__()

        if callbacks is None:
            callbacks = []

        if not isinstance(n_iter, int):
            raise ValueError("n_iter parameter must be an integer.")
        elif n_iter < 0:
            raise ValueError(f"n_iter must be >= 0, value: {n_iter}")

        if not isinstance(sliding_window, int):
            raise ValueError("sliding_window parameter must be an integer.")

        if not isinstance(min_delta, (int, float)):
            raise ValueError("min_delta parameter must be an integer or float.")

        self.n_iter = n_iter
        self.sliding_window = sliding_window
        self.min_delta = min_delta

        self.callbacks = []
        for callback in callbacks:
            self.addCallback(callback)

    # TODO: ensure that for patience of 1 million and 2 million there are the same ranks
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
        stop = False

        self.runCallbacks("before_simulation", strategy=self)

        for iter, error in enumerate(curve):
            if self.n_iter != 0 and iter >= self.n_iter:
                break
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
            elif error < best_error - self.min_delta:
                best_iter = iter
                best_error = error
                iter_wo_improvement = 0
                patience = self.patience(iter + 1)
            else:
                iter_wo_improvement += 1
                if iter_wo_improvement >= patience:
                    stop = True

            # after iteration callbacks
            self.runCallbacks(
                "after_iter",
                strategy=self,
                iter=iter,
                metric=error,
                iter_wo_improvement=iter_wo_improvement,
                patience=patience,
            )

            if stop:
                break

        self.runCallbacks("after_simulation", strategy=self)
        return best_iter, iter

    def addCallback(self, new_callback):
        from callbacks import IterativeStrategyCallback

        if not isinstance(new_callback, IterativeStrategyCallback):
            raise ValueError(f"Invalid callback={new_callback}")

        if self.callbacks:
            self.callbacks.append(new_callback)
        else:
            self.callbacks = [new_callback]

    def runCallbacks(self, method: str, **kwargs):
        # TODO: doesn't support stopping pased on ret value of callback
        # by using function for running callbacks like this
        if self.callbacks:
            for callback in self.callbacks:
                func = getattr(callback, method)
                if func(**kwargs):
                    break

    @property
    def patience(self) -> Callable[[int], int]:
        return lambda x: 0

    @classmethod
    def kwargs(cls) -> dict[str, str]:
        kwargs = super().kwargs()
        kwargs.update(
            {
                "n_iter": "n_iter",
                "sliding_window": "sw",
                "min_delta": "md",
            }
        )
        return kwargs
