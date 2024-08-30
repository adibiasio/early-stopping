from .polynomial_patience import PolynomialPatienceStrategy


class LinearPatienceStrategy(PolynomialPatienceStrategy):
    """
    Patience Equation: p(x) = ax + b
    """

    _name = "linear_patience"
    _short_name = "LP"

    def __init__(self, a: float | int = 0, b: float | int = 0, **kwargs):
        assert "degree" not in kwargs
        super().__init__(a=a, b=b, **kwargs)

    @classmethod
    def kwargs(cls) -> dict[str, str]:
        kwargs = super().kwargs()
        kwargs.pop("degree", None)
        return kwargs
