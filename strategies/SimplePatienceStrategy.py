from .IterativeStrategy import IterativeStrategy


class SimplePatienceStrategy(IterativeStrategy):
    """
    Patience Equation: p(x) = p
    """

    _name = "simple_patience"
    _short_name = "SP"

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
