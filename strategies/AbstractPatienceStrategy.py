from abc import abstractmethod
from typing import Callable

from .IterativeStrategy import IterativeStrategy


class AbstractPatienceStrategy(IterativeStrategy):

    def __init__(self, min_patience: int = 10, max_patience: int = 10000, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(min_patience, int):
            raise ValueError(f"Invalid parameter min_patience={min_patience}")

        if not isinstance(max_patience, int):
            raise ValueError(f"Invalid parameter max_patience={max_patience}")

        self.min_patience = min_patience
        self.max_patience = max_patience

    @property
    def patience(self) -> Callable[[int], int]:
        base_fn = self._patience_fn()

        def func(iter):
            p = base_fn(iter)
            return min(
                self.max_patience,
                max(self.min_patience, p),
            )

        return func

    @abstractmethod
    def _patience_fn(self) -> Callable[[int], int]:
        pass

    @classmethod
    def kwargs(cls) -> dict[str, str]:
        kwargs = super().kwargs()
        kwargs.update({
            "min_patience": "maxp",
            "max_patience": "minp",
        })
        return kwargs
