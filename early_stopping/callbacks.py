from abc import ABC

from .strategies.AbstractStrategy import AbstractStrategy
from .strategies.IterativeStrategy import IterativeStrategy

###########################################################
###################### Base Classes #######################
###########################################################


class StrategyCallback(ABC):
    """Interface for simulation strategy callbacks."""

    def __init__(self) -> None:
        pass

    def before_simulation(self, strategy: AbstractStrategy) -> None:
        """Run before simulation starts."""
        return

    def after_simulation(self, strategy: AbstractStrategy) -> None:
        """Run after simulation is finished."""
        return


class IterativeStrategyCallback(StrategyCallback):
    """Interface for iterative simulation callbacks."""

    def before_iter(
        self,
        strategy: IterativeStrategy,
        iter: int,
        metric: float,
        iter_wo_improvement: int,
        patience: int,
    ) -> bool:
        """Run before each iteration.  Return True when simulation should stop."""
        return False

    def after_iter(
        self,
        strategy: IterativeStrategy,
        iter: int,
        metric: float,
        iter_wo_improvement: int,
        patience: int,
    ) -> bool:
        """Run after each iteration.  Return True when simulation should stop."""
        return False


class SimulationCallback(ABC):
    """Interface for simulation callbacks."""

    has_save_artifacts: bool = False

    def __init__(
        self, strategy_callback: StrategyCallback, path: str | None = None
    ) -> None:
        self.strategy_callback = strategy_callback
        self.path = path

    def before_task(
        self,
        dataset: str,
        fold: int,
        curve_data: dict,
        strategies: dict,
        filters: dict,
    ):
        """Run before a task is processed."""

    def after_task(
        self,
        dataset: str,
        fold: int,
    ):
        """Run after a task is processed."""

    def before_strategy(
        self,
        model: str,
        error: float,
        eval_set: str,
        strategy: AbstractStrategy,
    ):
        """Run before the strategy simulation process."""

    def after_strategy(
        self,
        model: str,
        error: float,
        eval_set: str,
        strategy: AbstractStrategy,
    ):
        """Run after the strategy simulation process."""


###########################################################
################ Callback Implementations #################
###########################################################


class PatienceStrategyCallback(IterativeStrategyCallback):
    name = "Patience Curves"
    file_name = "patience_curves"

    def __init__(self, results: list) -> None:
        self.results = results
        self.x = "iter"
        self.y = self.x

    def before_iter(
        self,
        strategy: IterativeStrategy,
        iter: int,
        metric: float,
        iter_wo_improvement: int,
        patience: int,
    ) -> bool:
        if "iter" not in self.results:
            self.results["iter"] = []

        if "iter_wo_improvement" not in self.results:
            self.results["iter_wo_improvement"] = []

        if "patience" not in self.results:
            self.results["patience"] = []

        self.results["iter"].append(iter)
        self.results["iter_wo_improvement"].append(iter_wo_improvement)
        self.results["patience"].append(patience)

        return False


class LearningCurveStrategyCallback(IterativeStrategyCallback):
    name = "Learning Curves"
    file_name = "learning_curves"

    def __init__(self, results: list) -> None:
        self.results = results
        self.x = "iter"
        self.y = "metric"

    def before_iter(
        self,
        strategy: IterativeStrategy,
        iter: int,
        metric: float,
        iter_wo_improvement: int,
        patience: int,
    ) -> bool:
        if "iter" not in self.results:
            self.results["iter"] = []

        if "metric" not in self.results:
            self.results["metric"] = []

        self.results["iter"].append(iter)
        self.results["metric"].append(metric)

        return False


class GraphSimulationCallback(SimulationCallback):
    """
    Callback to generate curve graphs saved in the save_path directory.
    Adds significant overhead, so not recommended on large runs.
    """

    has_save_artifacts = True

    def __init__(
        self, strategy_callback: StrategyCallback, path: str | None = None
    ) -> None:
        self.strategy_callback = strategy_callback
        self.path = path
        self.figure = None
        self.axes = None
        self.legend = None
        self.results = None

    def before_task(
            self,
            dataset: str,
            fold: int,
            curve_data: dict,
            strategies: dict,
            filters: dict,
    ):
        import numpy as np
        from matplotlib import pyplot as plt

        total_curves = 0
        for model, data in curve_data.items():
            if filters["models"] and model not in filters["models"]:
                continue

            eval_sets, metrics, _ = data
            if filters["metrics"]:
                metrics = len(np.intersect1d(metrics, filters["metrics"]))
            else:
                metrics = len(metrics)

            if filters["eval_sets"]:
                eval_sets = len(np.intersect1d(eval_sets, filters["eval_sets"]))
            else:
                eval_sets = len(eval_sets)

            total_curves += metrics * eval_sets

        # more concise form, but doesn't account for filters
        # for _, data in curve_data.items():
        #     math.prod([len(arr) for arr in data[:2]])

        total_configs = 0
        for _, (_, configs) in strategies.items():
            total_configs += len(configs)
        num_axes = total_curves * total_configs  # total_simulations

        # Find a suitable layout (e.g., square root approach)
        cols = max(int(np.ceil(np.sqrt(num_axes))), 1)
        rows = max(int(np.ceil(num_axes / cols)), 1)

        # create figure and axes
        self.figure, _ = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        self.axes = self.generator(self.figure.axes)

    @staticmethod
    def generator(array):
        for item in array:
            yield item

    def after_task(self, dataset: str, fold: int):
        """
        Create a figure with subplots based on a list of Axes objects.
        """
        self.figure.suptitle(self.strategy_callback.name, fontsize=24, y=0.985)
        self.figure.legend(*self.legend)
        self.figure.tight_layout(pad=2.0)

        import os

        file_name = self.strategy_callback.file_name
        file_name = f"{file_name}_{dataset}_{fold}"
        # change path to self.path, filename.split(".")[0], filename-dataset-fold)?
        self.figure.savefig(os.path.join(self.path, file_name))

    def before_strategy(
        self, model: str, metric: str, eval_set: str, strategy: IterativeStrategy
    ):
        """Run before the strategy simulation process."""
        self.results = {}
        # TODO: maybe instead of always creating new callback, I can just update self.results field
        new_callback = self.strategy_callback(results=self.results)
        strategy.addCallback(new_callback=new_callback)

    def after_strategy(
        self, model: str, metric: str, eval_set: str, strategy: IterativeStrategy
    ):
        """Run after the strategy simulation process."""
        ax = next(self.axes)
        callback = strategy.callbacks.pop(0)
        self.plot_lines(callback.x, ax, y_label=callback.y, **self.results)

        if self.legend is None:
            self.legend = ax.get_legend_handles_labels()

        ax.set_title(f"{str(strategy)}\n{model}-{metric}-{eval_set}")

    def plot_lines(self, x: str, ax, y_label: str, **kwargs):
        if len(kwargs) < 2:
            raise ValueError("Not enough lines to construct line plot!")

        x_label = x
        x = kwargs[x_label]
        del kwargs[x_label]

        import seaborn as sns

        for label, line in kwargs.items():
            sns.lineplot(x=x, y=line, ax=ax, label=label)

        ax.legend().set_visible(False)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)
