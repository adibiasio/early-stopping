from typing import Callable

from strategies.PolynomialAdaptivePatienceStrategy import (
    PolynomialAdaptivePatienceStrategy,
)


class FeaturePatienceStrategy(PolynomialAdaptivePatienceStrategy):
    """
    Patience equation is influenced by features of the dataset currently being "trained" on.
    """

    _name = "feature_patience"
    _short_name = "FP"

    needs_curve_metadata = True

    def __init__(
        self,
        metadata: dict,
        min_rows: int = 10000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if "num_rows_train" not in metadata:
            raise ValueError(
                "Learning Curve Object does not have field 'num_rows_train'!"
            )
        num_rows_train = metadata["num_rows_train"]

        if not isinstance(num_rows_train, int):
            raise ValueError(
                f"Invalid parameter num_rows_train={num_rows_train} for strategy {self.name}"
            )

        if not isinstance(min_rows, int):
            raise ValueError(
                f"Invalid parameter min_rows={min_rows} for strategy {self.name}"
            )

        self.num_rows_train = num_rows_train
        self.min_rows = min_rows

    @property
    def patience(self) -> Callable[[int], int]:
        base_fn = super().patience

        def _patience_fn(iter):
            modifier = (
                1
                if self.num_rows_train <= self.min_rows
                else self.min_rows / self.num_rows_train
            )
            simple_early_stopping_rounds = max(
                round(modifier * self.max_patience),
                self.min_patience,
            )

            self.min_patience = simple_early_stopping_rounds

            return base_fn(iter)

        return _patience_fn

    def _base_name(self) -> str:
        return self._name

    @classmethod
    def user_params(cls) -> "set[str]":
        params = super().user_params()
        params.discard("degree")
        params = params.union(
            {"num_rows_train", "min_patience", "max_patience", "min_rows"}
        )
        return params
