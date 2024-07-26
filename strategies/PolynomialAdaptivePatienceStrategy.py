from typing import Callable

from strategies.AbstractPatienceStrategy import AbstractPatienceStrategy


class PolynomialAdaptivePatienceStrategy(AbstractPatienceStrategy):
    """
    Patience Equation: p(x) = ax^n + b
    """
    def __init__(self, a: float | int = 0, b: float | int = 0, degree: float | int = 1, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(a, (int, float)):
            raise ValueError("A value must be an int or float")

        if not isinstance(b, (int, float)):
            raise ValueError("B value must be an int or float")

        if not isinstance(degree, (int, float)):
            raise ValueError("Degree value must be an int or float")

        self.a = a
        self.b = b
        self.degree = degree


    def _patience_fn(self) -> Callable[[int], int]:
        def _get_patience(iter):
            return self.a * (iter ** self.degree) + self.b
        return _get_patience


    def _base_name(self) -> str:
        return "polynomial_adaptive_patience"


    @classmethod
    def user_params(cls) -> "set[str]":
        return super().user_params().union({ "a", "b", "degree" })
