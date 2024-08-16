from abc import ABC, abstractmethod


class AbstractStrategy(ABC):
    _name = ""
    _short_name = ""

    needs_curve_metadata = False

    simulate_columns = [
        "total_iter",
        "best_iter",
        "opt_iter",
        "best_error",
        "opt_error",
        "error_diff",
        "percent_error_diff",
        "percent_iter_diff",
    ]

    def simulate(
        self, stopping_curve: list[float], eval_curve: list[float]
    ) -> list[int | float]:
        """
        Returns 1-indexed iterations
        """
        chosen_iter, total_iter = self._run(stopping_curve)
        opt_iter = eval_curve.index(min(eval_curve))
        error_diff = eval_curve[chosen_iter] - eval_curve[opt_iter]
        max_eval = max(eval_curve)
        percent_error_diff = error_diff / (max_eval if max_eval != 0 else 1)
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
        Parameters:
        --------------
        curve:


        Return:
        --------------
        best iteration (zero indexed)

        total iterations (zero indexed)
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
                print(f"Warning: '{key}' is not a valid parameter for strategy={self.name}.")

# TODO: refactor all _short_kwargs into kwargs()
# TODO: ensure simple patience user_params uses patience instead of b


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
            to their two-character abbreviations.
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
