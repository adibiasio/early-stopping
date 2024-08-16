from strategies.PolynomialAdaptivePatienceStrategy import (
    PolynomialAdaptivePatienceStrategy,
)


class LinearAdaptivePatienceStrategy(PolynomialAdaptivePatienceStrategy):
    """
    Patience Equation: p(x) = ax + b
    """

    _name = "linear_adaptive_patience"
    _short_name = "LP"

    def __init__(self, a: float | int = 0, b: float | int = 0, **kwargs):
        super().__init__(a=a, b=b, **kwargs)

    @classmethod
    def kwargs(cls) -> dict[str, str]:
        kwargs = super().kwargs()
        kwargs.pop("degree", None)
        return kwargs
