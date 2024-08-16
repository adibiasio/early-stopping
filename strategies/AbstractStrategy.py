from abc import ABC, abstractmethod


class AbstractStrategy(ABC):
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

    _name = ""
    _short_name = ""
    _short_kwargs = {}

    needs_curve_metadata = False

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

    def update_params(self, **kwargs):
        """
        Update instance parameters based on the provided keyword arguments.

        Parameters:
        -----------
        **kwargs: keyword arguments where the key is the parameter name
                  and the value is the new value for that parameter.
        """
        instance_vars = list(vars(self).keys())
        properties = [
            name
            for name in dir(self)
            if isinstance(getattr(type(self), name, None), property)
        ]
        attributes = instance_vars + properties
        for key, value in kwargs.items():
            if key in attributes:
                setattr(self, key, value)
            else:
                print(f"Warning: '{key}' is not an attribute of the instance.")

    # @classmethod
    # def init_short_kwargs(cls, parent_class, new_short_kwargs):
    #     _short_kwargs = parent_class._short_kwargs.copy()
    #     _short_kwargs.update(
    #         {
    #             "sliding_window": "sw",
    #             "min_delta": "md",
    #         }
    #     )
    #     return _short_kwargs

    @property
    def name(self):
        return self._name

    @classmethod
    def user_params(cls) -> "set[str]":
        return set()

    def __str__(self) -> str:
        params = self.user_params()
        short_params = self._short_kwargs

        result = [self._short_name]
        for param in params:
            result.append(f"{short_params[param]}={round(getattr(self, param),3)}")

        return ";".join(result)
