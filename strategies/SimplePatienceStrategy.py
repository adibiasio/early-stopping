from strategies.LinearAdaptivePatienceStrategy import LinearAdaptivePatienceStrategy


class SimplePatienceStrategy(LinearAdaptivePatienceStrategy):
    """
    Patience Equation: p(x) = p
    """

    _name = "simple_patience"
    _short_name = "SP"
    _short_kwargs = LinearAdaptivePatienceStrategy._short_kwargs.copy()
    _short_kwargs.update(
        {
            "b": "p",
        }
    )

    def __init__(self, patience: int = 0, **kwargs):
        super().__init__(b=patience, **kwargs)

    @property
    def patience(self):
        return super().patience

    @patience.setter
    def patience(self, new_patience):
        self.b = new_patience

    @classmethod
    def user_params(cls) -> "set[str]":
        params = super().user_params()
        params.discard("a")
        return params
