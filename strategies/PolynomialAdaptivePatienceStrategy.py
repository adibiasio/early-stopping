from typing import Callable

from strategies.AbstractPatienceStrategy import AbstractPatienceStrategy


class PolynomialAdaptivePatienceStrategy(AbstractPatienceStrategy):
    """
    Patience Equation: p(x) = a * x^degree + b
    """

    _name = "polynomial_adaptive_patience"
    _short_name = "PP"
    _short_kwargs = AbstractPatienceStrategy._short_kwargs.copy()
    _short_kwargs.update(
        {
            "a": "a",
            "b": "b",
            "degree": "n",
        }
    )

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
        return self._base_class()._name

    def _base_class(self) -> AbstractPatienceStrategy:
        from strategies.LinearAdaptivePatienceStrategy import (
            LinearAdaptivePatienceStrategy,
        )
        from strategies.SimplePatienceStrategy import SimplePatienceStrategy

        if self.degree == 1:
            if self.a == 0:
                return SimplePatienceStrategy
            return LinearAdaptivePatienceStrategy
        return self.__class__

    def __str__(self) -> str:
        base_class = self._base_class()
        params = base_class.user_params()
        short_params = base_class._short_kwargs

        result = []
        for param in params:
            result.append(f"{short_params[param]}={round(getattr(self, param),3)}")

        result.sort()
        result.insert(0, self._base_name())

        return ";".join(result)

    @classmethod
    def user_params(cls) -> "set[str]":
        return super().user_params().union({"a", "b", "degree"})
