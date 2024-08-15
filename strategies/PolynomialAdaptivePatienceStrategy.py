from typing import Callable

from strategies.AbstractPatienceStrategy import AbstractPatienceStrategy

# TODO: also test adding min/max patience parameter values for these strategies
# (refer to autogluon patience strategy)


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
        min_patience: int = 10,
        max_patience: int = 10000,
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
        self.min_patience = min_patience
        self.max_patience = max_patience

    @property
    def patience(self) -> Callable[[int], int]:
        def _patience_fn(iter):
            p = round(self.a * (iter**self.degree) + self.b)
            return min(
                self.max_patience,
                max(self.min_patience, p),
            )

        return _patience_fn

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
