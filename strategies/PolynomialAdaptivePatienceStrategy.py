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

    def base_class(self) -> AbstractPatienceStrategy:
        from strategies.LinearAdaptivePatienceStrategy import (
            LinearAdaptivePatienceStrategy,
        )
        from strategies.SimplePatienceStrategy import SimplePatienceStrategy

        if self.a == 0:
            return SimplePatienceStrategy

        if self.degree == 1:
            return LinearAdaptivePatienceStrategy

        return self.__class__

    def _base_name(self) -> str:
        return self.base_class()._name

    def __str__(self) -> str:
        """
        Returns:
        --------
        str:
            A single line string listing a strategy and its configurations
            with abbreviated strategy/parameter names and rounded values.
        """
        # TODO: make use of abstract strategy __str__ method
        # this is code dupe, but not sure of a good way to reuse code here
        result = []
        base_class = self.base_class()
        for param, short in base_class.kwargs().items():
            val = self.b if param == "patience" else getattr(self, param)
            result.append(f"{short}={round(val,3)}")

        result.sort()
        result.insert(0, base_class._short_name)
        return ";".join(result)


    # TODO: currently, if poly patience is actually simple patience
    # kwargs still returns "a" and "degree", which it shouldn't
    @classmethod
    def kwargs(cls) -> dict[str, str]:
        kwargs = super().kwargs()
        kwargs.update(
            {
                "a": "a",
                "b": "b",
                "degree": "n",
            }
        )
        return kwargs
