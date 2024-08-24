from strategies.IterativeStrategy import IterativeStrategy


class SimplePatienceStrategyV2(IterativeStrategy):
    """
    Patience Equation: p(x) = p
    """

    _name = "simple_patience_v2"
    _short_name = "SPv2"

    def __init__(self, b: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.b = b

    @property
    def patience(self):
        """
        Redefined here because cannot change property
        setter without also redefining property in same class.
        """
        return lambda x: self.b

    @classmethod
    def kwargs(cls) -> dict[str, str]:
        kwargs = super().kwargs()
        kwargs.update(
            {
                "b": "b",
            }
        )
        return kwargs
