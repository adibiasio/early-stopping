from typing import List
from strategies.AbstractStrategy import AbstractStrategy

class FixedIterationStrategy(AbstractStrategy):
    def __init__(self, n_iter: int = 10):
        super().__init__()

        if not isinstance(n_iter, int):
            raise ValueError("Valid number of iterations must be specified to use a Fixed Iteration Stopping Strategy")

        self.n_iter = n_iter


    def _run(self, curve: List[float]):
        if self.n_iter > len(curve):
            best_iter = len(curve) - 1
            total_iter = best_iter
            # raise ValueError("Length of Curve is longer than number of iterations set for Fixed Iteration Stopping Strategy")

        best_iter = self.n_iter
        total_iter = self.n_iter

        return best_iter, total_iter


    @property
    def name(self):
        return "fixed_iterations"


    @classmethod
    def user_params(cls) -> "set[str]":
        return super().user_params().union({ "n_iter" })
