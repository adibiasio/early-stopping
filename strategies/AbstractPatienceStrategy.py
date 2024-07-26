from abc import abstractmethod
from typing import Callable

from strategies.IterativeStrategy import IterativeStrategy


class AbstractPatienceStrategy(IterativeStrategy):
    def __init__(self, **kwargs):
        """
        Parameters:
        --------------
        sliding
            window size for averaging last n errors

        min_delta
            min difference in error to be classified as model relapse

        """
        super().__init__(patience = self._patience_fn(), **kwargs)
    

    @abstractmethod
    def _patience_fn(self) -> Callable[[int], int]:
        pass
