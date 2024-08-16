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
        # TODO: i think b is supposed to be set to initial setting of min_patience here
        # but i refactored stuff, so it got messed up
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

    def _patience_fn(self) -> Callable[[int], int]:
        base_fn = super()._patience_fn

        def func(iter):
            modifier = (
                1
                if self.num_rows_train <= self.min_rows
                else self.min_rows / self.num_rows_train
            )
            self.min_patience = max(
                round(modifier * self.max_patience),
                self.min_patience,
            )

            return base_fn(iter)

        return func

    # @property
    # def patience(self) -> Callable[[int], int]:
    #     base_fn = super().patience

    #     def _patience_fn(iter):
    #         modifier = (
    #             1
    #             if self.num_rows_train <= self.min_rows
    #             else self.min_rows / self.num_rows_train
    #         )
    #         self.min_patience = max(
    #             round(modifier * self.max_patience),
    #             self.min_patience,
    #         )

    #         return base_fn(iter)

    #     return _patience_fn

    def _base_name(self) -> str:
        return self._name

    # TODO: figure out how to handle num_rows_train
    # because you don't set it directly currently
    # (you pass in metadata object and use that to 
    # set num_rows_train)
    @classmethod
    def kwargs(cls) -> dict[str, str]:
        kwargs = super().kwargs()
        kwargs.update({
            "num_rows_train": "rt",
            "min_rows": "mr",
        })
        return kwargs
