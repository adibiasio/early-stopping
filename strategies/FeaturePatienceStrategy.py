from __future__ import annotations

from .PolynomialAdaptivePatienceStrategy import (
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
        max_offset: int = 300,
        min_offset: int | None = None,
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

        self.max_offset = max_offset
        self.num_rows_train = num_rows_train
        self.min_rows = min_rows

        if min_offset is not None:
            self.min_offset = min_offset
        else:
            self.min_offset = self.b

        modifier = (
            1
            if self.num_rows_train <= self.min_rows
            else self.min_rows / self.num_rows_train
        )
        self.min_patience = max(
            round(modifier * self.max_offset),
            self.min_offset,
        )

    @classmethod
    def kwargs(cls) -> dict[str, str]:
        kwargs = super().kwargs()
        kwargs.update(
            {
                "num_rows_train": "rt",
                "min_rows": "mr",
                "max_offset": "max_o",
                "min_offset": "min_o",
            }
        )
        return kwargs
