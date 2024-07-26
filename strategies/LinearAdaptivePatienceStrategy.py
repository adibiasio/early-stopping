from strategies.PolynomialAdaptivePatienceStrategy import PolynomialAdaptivePatienceStrategy

class LinearAdaptivePatienceStrategy(PolynomialAdaptivePatienceStrategy):
    """
    Patience Equation: p(x) = ax + b
    """
    def __init__(self, a: float | int = 0, b: float | int = 10, **kwargs):
        super().__init__(a=a, b=b, degree=1, **kwargs)


    def _base_name(self) -> str:
        return "linear_adaptive_patience"


    @classmethod
    def user_params(cls) -> "set[str]":
        params = super().user_params()
        params.remove("degree")
        return params
