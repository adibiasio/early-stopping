from strategies.LinearAdaptivePatienceStrategy import LinearAdaptivePatienceStrategy


class SimplePatienceStrategy(LinearAdaptivePatienceStrategy):
    """
    Patience Equation: p(x) = p
    """

    _name = "simple_patience"
    _short_name = "SP"

    def __init__(self, patience: int = 0, **kwargs):
        super().__init__(b=patience, **kwargs)

    @property
    def patience(self):
        """
        Redefined here because cannot change property
        setter without also redefining property in same class.
        """
        return super().patience

    @patience.setter
    def patience(self, new_patience: int):
        """
        Note that patience is alias for b in 
        the polynomial patience equation.

        Parameters:
        ----------
        new_patience: int
            The new value for patience.
        """
        self.b = new_patience

    @classmethod
    def kwargs(cls) -> dict[str, str]:
        kwargs = super().kwargs()
        kwargs.pop("a", None)
        kwargs.pop("b", None)
        kwargs.update({
            "patience": "p",
        })
        return kwargs
