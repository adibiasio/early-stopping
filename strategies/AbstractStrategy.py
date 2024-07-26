from typing import List, Tuple
from abc import ABC, abstractmethod


class AbstractStrategy(ABC):
    def __init__(self):
        pass


    def simulate(self, stopping_curve: List[float], eval_curve: List[float]):
        """
        Returns 1-indexed iterations
        """
        chosen_iter, total_iter = self._run(stopping_curve)
        opt_iter = eval_curve.index(min(eval_curve))
        error_diff = eval_curve[chosen_iter] - eval_curve[opt_iter]
        percent_error_diff = error_diff / max(eval_curve)
        percent_iter_diff = (total_iter - opt_iter) / (opt_iter if opt_iter != 0 else 1)
        return [total_iter + 1, chosen_iter + 1, opt_iter + 1, eval_curve[chosen_iter], eval_curve[opt_iter], error_diff, percent_error_diff, percent_iter_diff]


    @abstractmethod
    def _run(self, curve: List[float]) -> Tuple[int, int]:
        """
        Parameters:
        --------------
        curve: 


        Return:
        --------------
        best iteration (zero indexed)

        total iterations (zero indexed)
        """
        pass


    @property
    @abstractmethod
    def name(self):
        pass


    @classmethod
    def user_params(cls) -> "set[str]":
        return set()
