from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .abstract_strategy import AbstractStrategy

if TYPE_CHECKING:
    from ..callbacks import IterativeStrategyCallback


class IterativeStrategy(AbstractStrategy):
    """
    Framework for a strategy that determines when to stop by iterating
    over the values of the learning curve.

    Parameters:
    --------------
    time_per_iter: float
        The time in seconds per iteration that the model took to fit.
        Used by `time_limit`.
        Will be provided to the class by the factory since `cls.needs_time_per_iter == True`.

    n_iter: int, default = 0
        Maximum number of iterations to train for. Unlimited if 0.

    sliding_window: int, default = 1
        Window size for averaging last n errors.

    min_delta: float | int, default = 0
        Minimum decrease in error to qualify as an improvement in model performance.

    time_limit: float, default = -1
        `time_limit` corresponds to the time in seconds allowed for fitting the model (simulated).
        For example, if a model fit 1000 iterations in 500 seconds, and `time_limit=100`,
        then the stopping strategy will be forced to stop at iteration 200 since it ran out of time.
        (200 iterations would take 100 seconds).
        Ignored if -1.

    callbacks: list[IterativeStrategyCallback], default = []
        List of IterativeStrategyCallback objects, whose methods will be called throughout the
        simulation process.
    """

    needs_time_per_iter = True

    def __init__(
        self,
        time_per_iter: float,
        n_iter: int = 0,
        sliding_window: int = 1,
        min_delta: float | int = 0,
        time_limit: float = -1,
        callbacks: list[IterativeStrategyCallback] = [],
    ):
        super().__init__()

        if not isinstance(time_per_iter, (int, float)):
            raise ValueError("time_per_iter parameter must be an integer or float.")

        if not isinstance(n_iter, int):
            raise ValueError("n_iter parameter must be an integer.")
        elif n_iter < 0:
            raise ValueError(f"n_iter must be >= 0, value: {n_iter}")

        if not isinstance(sliding_window, int):
            raise ValueError("sliding_window parameter must be an integer.")

        if not isinstance(min_delta, (int, float)):
            raise ValueError("min_delta parameter must be an integer or float.")

        if not isinstance(time_limit, (int, float)):
            raise ValueError("time_limit parameter must be an integer or float.")

        self.time_per_iter = time_per_iter
        self.n_iter = n_iter
        self.sliding_window = sliding_window
        self.min_delta = min_delta
        self.time_limit = time_limit

        self.callbacks = []
        for callback in callbacks:
            self.add_callback(callback)

    # TODO: ensure that for patience of 1 million and 2 million there are the same ranks
    def _run(self, curve: list[float]):
        """
        Traces through the learning curve and decides when to stop.

        Parameters:
        -----------
        curve: list[float]
            A list of performance metrics calculated at each iteration
            of the training process.

        Returns:
        --------
        chosen_iter: int (zero indexed)
            The iteration chosen as the "best" iteration (lowest observed error)
            according to the stopping strategy and its parameters.
        total iter: int (zero indexed)
            The total number of iterations the strategy ran before stopping.
        """
        best_iter = 0
        best_error = None
        sliding_sum = 0
        iter_wo_improvement = 0
        patience = self.patience(0)
        stop = False

        self.run_callbacks("before_simulation", strategy=self)

        for iter, error in enumerate(curve):
            if self.n_iter != 0 and iter >= self.n_iter:
                break
            stop = stop or self.run_callbacks(
                "before_iter",
                strategy=self,
                iter=iter,
                metric=error,
                iter_wo_improvement=iter_wo_improvement,
                patience=patience,
            )

            if stop:
                break

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
                patience = self.patience(iter)
            else:
                iter_wo_improvement += 1
                if iter_wo_improvement >= patience:
                    stop = True

            if self.time_limit != -1:
                time_spent = iter * self.time_per_iter
                if time_spent >= self.time_limit:
                    stop = True

            stop = stop or self.run_callbacks(
                "after_iter",
                strategy=self,
                iter=iter,
                metric=error,
                iter_wo_improvement=iter_wo_improvement,
                patience=patience,
            )

            if stop:
                break

        self.run_callbacks("after_simulation", strategy=self)
        return best_iter, iter

    def add_callback(self, new_callback):
        from ..callbacks import IterativeStrategyCallback

        if not isinstance(new_callback, IterativeStrategyCallback):
            raise ValueError(f"Invalid callback={new_callback}")

        if self.callbacks:
            self.callbacks.append(new_callback)
        else:
            self.callbacks = [new_callback]

    def run_callbacks(self, method: str, **kwargs) -> True:
        stop = False
        if self.callbacks:
            for callback in self.callbacks:
                func = getattr(callback, method)
                stop = stop or func(**kwargs)
        return stop

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
