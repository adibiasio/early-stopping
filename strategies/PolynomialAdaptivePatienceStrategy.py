from typing import Callable

from .AbstractPatienceStrategy import AbstractPatienceStrategy


class PolynomialAdaptivePatienceStrategy(AbstractPatienceStrategy):
    """
    Patience Equation: p(x) = a * x^degree + b
    """

    _name = "polynomial_adaptive_patience"
    _short_name = "PP"

    def __init__(
        self,
        a: float | int = 0,
        b: int = 0,
        degree: float | int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not isinstance(a, (int, float)):
            raise ValueError("A value must be an int or float")

        if not isinstance(b, int):
            raise ValueError("B value must be an int")

        if not isinstance(degree, (int, float)):
            raise ValueError("Degree value must be an int or float")

        self.a = a
        self.b = b
        self.degree = degree

    def _patience_fn(self) -> Callable[[int], int]:
        def func(iter):
            return round(self.a * (iter**self.degree) + self.b)

        return func

    def _base_name(self) -> str:
        return self._name

    @classmethod
    def kwargs(cls) -> dict[str, str]:
        kwargs = super().kwargs()
        kwargs.update(
            {
                "a": "a",
                "b": "b",
                "degree": "degree",
            }
        )
        return kwargs
