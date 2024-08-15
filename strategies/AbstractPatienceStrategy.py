from abc import abstractmethod
from typing import Callable

from strategies.IterativeStrategy import IterativeStrategy


class AbstractPatienceStrategy(IterativeStrategy):

    @property
    @abstractmethod
    def patience(self) -> Callable[[int], int]:
        return
