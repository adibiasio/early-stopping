from strategies.PolynomialAdaptivePatienceStrategy import PolynomialAdaptivePatienceStrategy

class SimplePatienceStrategy(PolynomialAdaptivePatienceStrategy):
    """
    Patience Equation: p(x) = p
    """
    def __init__(self, patience: int = 10, **kwargs):
        super().__init__(a=0, b=patience, **kwargs)


    def _base_name(self) -> str:
        return "simple_patience"


    @classmethod
    def user_params(cls) -> "set[str]":
        params = super().user_params()
        params.discard("a")
        params.discard("b")
        params.discard("degree")
        params.add("patience")
        return params
