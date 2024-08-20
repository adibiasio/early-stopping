from strategies.FeaturePatienceStrategy import FeaturePatienceStrategy


class AutoGluonStrategy(FeaturePatienceStrategy):
    """
    Patience equation is influenced by features of the dataset currently being "trained" on.
    """

    _name = "autogluon_patience"
    _short_name = "AG"

    # default parameter settings within autogluon
    defaults = {
        "a": 0.3,
        "b": 10,
        "min_patience": 20,
        "max_patience": 300,
    }

    def __init__(
        self,
        metadata: dict,
        simple: int = 0,
        **kwargs,
    ):
        self.simple = simple
        defaults = self.defaults.copy()
        del defaults["a"]

        super().__init__(metadata, **defaults, **kwargs)

    def _base_name(self) -> str:
        from strategies.SimplePatienceStrategy import SimplePatienceStrategy

        base_class = self.base_class()
        prefix = "simple_" if base_class == SimplePatienceStrategy else "adaptive_"
        return prefix + super()._base_name()

    @property
    def simple(self):
        from strategies.SimplePatienceStrategy import SimplePatienceStrategy

        base_class = self.base_class()
        return 0 if base_class == SimplePatienceStrategy else 1

    @simple.setter
    def simple(self, new_value: int):
        if new_value == 0:
            self.a = 0
        elif new_value == 1:
            self.a = self.defaults["a"]
        else:
            raise ValueError(
                f"Invalid value {new_value} for simple parameter. Set = 0 to specify AG simple patience or = 1 to specify AG adaptive patience"
            )

    @classmethod
    def kwargs(cls) -> dict[str, str]:
        kwargs = super().kwargs()
        kwargs.update(
            {
                "simple": "s",
            }
        )
        return kwargs
