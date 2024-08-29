from abc import ABC, abstractmethod


class AbstractStrategy(ABC):
    """
    Abstract Class serving as a framework for early stopping strategy classes.
    Interface with a stopping strategy by invoking the simulate() method on
    your desired learning curves.
    
    Attributes:
    ----------
    name: str
        The name of this stopping strategy.
    needs_curve_metadata: bool
        Whether this strategy needs the curve metadata object.
    needs_time_per_iter: bool
        Whether this strategy needs the time taken per iteration during training.
    """
    _name = ""
    _short_name = ""

    needs_curve_metadata = False
    needs_time_per_iter = False

    simulate_columns = [
        "total_iter",
        "chosen_iter",
        "opt_iter",
        "chosen_error",
        "opt_error",
        "error_diff",
        "percent_error_diff",
        "percent_iter_diff",
    ]

    def simulate(
        self, stopping_curve: list[float], eval_curve: list[float]
    ) -> list[int | float]:
        """
        Applies stopping strategy to inputted learning curves and
        calculates strategy performance metrics.

        Parameters:
        -----------
        stopping_curve: list[float]
            A list of performance metrics calculated on the validation set
            at each iteration of the training process. This is the curve that
            the stopping strategy logic gets applied to.
        eval_curve: list[float]
            A list of performance metrics calculated at some other evaluation set
            (i.e. test or train) at each iteration of the training process. This
            is the curve used to evaluate the performance of the stopping strategy.

        Returns:
        --------
        total_iter: int
            The total number of iterations the strategy ran before stopping.
        chosen_iter: int
            The iteration chosen by the strategy with the lowest observed error.
            In other words, the iteration within the range (0, total_iter) with the lowest error.
        opt_iter: int
            The iteration with the lowest error across all available iterations.
            In other words, eval_curve.index(min(eval_curve))
        chosen_error: float
            The error value at chosen_iter. Equivalent to: eval_curve[chosen_iter]
        opt_error: float
            The error value at opt_iter. Equivalent to: eval_curve[opt_iter]
        error_diff: float
            The difference in error between opt_error and chosen_error.
        percent_error_diff: float
            The percent error reduction of the optimal compared to the chosen.
            Note that this metric should be bounded by 0 and 1.
        percent_iter_diff: float
            The percent difference between the total number of iterations ran
            and the optimal stopping iteration.
        """
        chosen_iter, total_iter = self._run(stopping_curve)
        opt_iter = eval_curve.index(min(eval_curve))
        error_diff = eval_curve[chosen_iter] - eval_curve[opt_iter]
        assert error_diff >= 0
        if eval_curve[opt_iter] == eval_curve[chosen_iter]:
            percent_error_diff = 0
        else:
            percent_error_diff = 1 - (eval_curve[opt_iter] / (eval_curve[chosen_iter] if eval_curve[chosen_iter] != 0 else 1))
        percent_iter_diff = (total_iter - opt_iter) / (opt_iter if opt_iter != 0 else 1)
        return [
            total_iter + 1,
            chosen_iter + 1,
            opt_iter + 1,
            eval_curve[chosen_iter],
            eval_curve[opt_iter],
            error_diff,
            percent_error_diff,
            percent_iter_diff,
        ]

    @abstractmethod
    def _run(self, curve: list[float]) -> tuple[int, int]:
        """
        Traces through the learning curve and decides when to stop.

        Parameters:
        -----------
        curve: list[float]
            A list of performance metrics calculated at each iteration
            of the training process.

        Returns:
        --------
        chosen_iter: int (zero indexed)
            The iteration chosen as the "best" iteration (lowest observed error)
            according to the stopping strategy and its parameters.
        total iter: int (zero indexed)
            The total number of iterations the strategy ran before stopping.
        """
        pass

    def update_params(self, **kwargs) -> None:
        """
        Updates any valid strategy parameter with a new value.
        Note that a valid strategy parameter is any parameter
        listed in self.user_params().

        Parameters:
        -----------
        **kwargs: keyword arguments where the key is the parameter name
                  and the value is the new value for that parameter.
        """
        user_params = self.user_params()
        for key, value in kwargs.items():
            if key in user_params:
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"'{key}' is not a valid parameter for strategy={self.name}."
                )

    @property
    def name(self):
        """
        Returns:
        --------
        str:
            The name of this strategy.
        """
        return self._name

    @classmethod
    def kwargs(cls) -> dict[str, str]:
        """
        Returns:
        --------
        dict[str, str]:
            Mapping of all valid parameters for this strategy
            to their parameter abbreviations.
        """
        return {}

    @classmethod
    def user_params(cls) -> set[str]:
        """
        Returns:
        --------
        set[str]:
            All valid parameters for this strategy.
        """
        return set(cls.kwargs().keys())

    def __str__(self) -> str:
        """
        Returns:
        --------
        str:
            A single line string listing a strategy and its configurations
            with abbreviated strategy/parameter names and rounded values.
        """
        result = []
        for param, short in self.kwargs().items():
            result.append(f"{short}={round(getattr(self, param),3)}")

        result.sort()
        result.insert(0, self._short_name)
        return ";".join(result)
