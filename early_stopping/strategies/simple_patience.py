from .linear_patience import LinearPatienceStrategy


class SimplePatienceStrategy(LinearPatienceStrategy):
    """
    Patience Equation: p(x) = p
    """

    _name = "simple_patience"
    _short_name = "SP"

    def __init__(self, b: int = 0, **kwargs):
        super().__init__(b=b, **kwargs)

    @classmethod
    def kwargs(cls) -> dict[str, str]:
        kwargs = super().kwargs()
        kwargs.pop("a", None)
        return kwargs
