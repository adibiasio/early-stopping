import itertools
import math
import os
import random
from typing import Callable

import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from callbacks import SimulationCallback
from strategies.AbstractStrategy import AbstractStrategy
from strategies.StrategyFactory import StrategyFactory
from utils.load import load_json
from utils.s3_utils import download_folder, is_s3_url


class StoppingSimulator:
    data_dir_name = "data"
    output_dir_name = "SimulatorRuns"

    def __init__(
        self,
        search_method: str = "random",
        callbacks: list[SimulationCallback] = None,
        output_dir: str = output_dir_name,
        seed: int = 42,
    ):
        self.factory = StrategyFactory()
        self.simulations = None
        self.tasks = []
        self.ranks = {}
        self.callbacks = []

        self._validate_and_preprocess_init_kwargs(
            search_method=search_method,
            callbacks=callbacks,
            output_dir=output_dir,
            seed=seed,
        )

        # 600 configs
        self.default_strategies = {
            "simple_patience": {"patience": (1, 50, 1)},
            "linear_adaptive_patience": {"a": (0.01, 0.51, 0.05), "b": (1, 50, 1)},
        }

    def load_curves(
        self,
        path: str,
        suffix: str = ".json",
        save_path: str | None = None,
        append: bool = True,
    ) -> list[str]:
        """
        Retrieves paths to all learning curve files in directory specified by 'path'.
        If path points to s3, downloads all curve files to local storage and
        tracks local paths.

        Parameters:
        -----------
        path: str
            local or s3 path to root directory containing learning curve .json files in
            the following format:
                .
                ├── dataset_a/
                │   ├── 0/
                │   │   └── learning_curves.json
                │   ├── 1/
                │   │   └── learning_curves.json
                │   └── 2/
                │       └── learning_curves.json
                ├── dataset_b/
                │   ├── 0/
                │   │   └── learning_curves.json
                │   ├── 1/
                │   │   └── learning_curves.json
                │   └── 2/
                │       └── learning_curves.json
                └── ...
        suffix: str
            the common suffix for all learning curve files, i.e. .../learning_curves.json
            would have a common suffix of 'learning_curves.json' if all curve files were
            named the same.
        save_path: str
            where all of the loaded curve files should be stored locally
            by default they are stored in the data/ folder of the current run's
            output dir.
        append: bool
            Whether to append to or overwrite the current curve tasks. Default = True

        Returns:
        --------
        List[str]: list of local paths to all learning curve files to be included in the simulation
        """
        import glob

        if is_s3_url(path):
            if not save_path:
                save_path = os.path.join(self.output_dir, self.data_dir_name)
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            download_folder(path=path, local_dir=save_path)
            path = save_path

        if not os.path.isdir(path):
            raise ValueError(f"The path '{path}' is not a valid directory.")
        paths = glob.glob(
            os.path.join(path, f"**/*{suffix}" if suffix else "*"), recursive=True
        )

        if append:
            self.tasks.extend(paths)
        else:
            self.tasks = paths

        return self.tasks

    def rank(
        self,
        by: str = "error_then_iter",
        eval_sets: str | list[str] | None = ["val", "test"],
        use_cache: bool = False,
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        """
        Ranks strategy simulation results.

        This method will run simulations if the run() method was not previously called or
        the use_cache flag is not set to True.
        Access rank dataframes using the getRanks() method.

        Parameters:
        -----------
        by: str, default = "error_then_iter"
            The method used to rank sstrategy simulation results. Possible methods include:
                error:
                    Rank strategies by differences between optimal error for a learning curve
                    and the best error identified, regardless of the number of iterations it took.
                iter:
                    Rank strategies by differences between the number of iterations ran before
                    stopping, regardless of performance.
                error_then_iter: (default)
                    Rank strategies by differences between optimal error for a learning curve
                    and the best error identified, then by the number of iterations it took.
                iter_then_error:
                    Rank strategies by differences between the number of iterations ran before
                    stopping, then by differences in error.
        eval_sets: str | list[str] | None, default = ["val", "test"],
            Filter for which eval_sets of the learning curves should be ranked on and used during simulation.
        use_cache: bool, default = False
            If the run() method has been called before (i.e. simulator has a simulations dataframe),
            and use_cache is set to True, the pre-existing simulations dataframe will be used for ranking
            purposes and the run() method will not be called.
        **kwargs:
            Any additional valid parameters listed for the run() method can be passed into rank(),
            and will be subsequently passed into run().

        Returns:
        --------
        dict[str, pd.DataFrame]:
            A dictionary mapping eval_sets to their rank dataframes.
        """
        if not use_cache or self.simulations is None:
            self.run(eval_sets=eval_sets, **kwargs)

        eval_sets = self._validate_and_preprocess_filter(filter=eval_sets)

        self.ranks = {}
        for eval_set in eval_sets:
            self.ranks[eval_set] = self._rank(eval_set=eval_set, by=by)
            self.ranks[eval_set].to_csv(
                os.path.join(self.output_dir, f"{eval_set}-ranks.csv"),
                index=False,
            )

        return self.ranks

    def run(
        self,
        strategies: dict | None = None,
        models: str | list[str] | None = None,
        metrics: str | list[str] | None = None,
        eval_sets: str | list[str] | None = None,
        mode: str = "ray",
    ) -> pd.DataFrame:
        """
        Runs simulations across all strategies.

        Parameters:
        -----------
        strategies: dict | None, default = None
            A dictionary mapping strategies to their parameter configuration lists.
            More information on the format of strategies can be found in self._preprocess_strategies()
        models: str | list[str] | None, default = None
            Filter for which models of the learning curves should be used during simulation.
        metrics: str | list[str] | None, default = None
            Filter for which metrics of the learning curves should be used during simulation.
        eval_sets: str | list[str] | None, default = None
            Filter for which eval_sets of the learning curves should be used during simulation.
        mode: str, default = "ray"
            Can be one of ["seq", "ray"]
            If "seq", tasks will be run sequentially
            If "ray", tasks will be run in parallel via ray.

        Returns:
        --------
        pd.DataFrame:
            The pandas dataframe containing all simulation results across the filtered learning curves.
        """
        if strategies is None:
            strategies = self.default_strategies
        strategies = self._preprocess_strategies(strategies=strategies)

        filters = dict(
            models=self._validate_and_preprocess_filter(filter=models),
            metrics=self._validate_and_preprocess_filter(filter=metrics),
            eval_sets=eval_sets,
        )

        kwargs = dict(
            strategies=strategies,
            filters=filters,
        )

        if mode == "ray":
            results = self._process_tasks_parallel(**kwargs)
        else:
            results = self._process_tasks_seq(**kwargs)

        self.simulations = pd.concat(results)
        self.simulations.to_csv(os.path.join(self.output_dir, f"simulations.csv"))

        return self.simulations

    def getRanks(self, eval_set: str) -> pd.DataFrame:
        """
        Retrieves rank dataframe for specified eval_set.

        Parameters:
        -----------
        eval_set: str
            The name of the eval_set to retrieve ranks for.

        Returns:
        --------
        pd.DataFrame:
            The ranks dataframe for eval_set.
        """
        if not self.ranks:
            raise RuntimeError(
                "Simulator has no rankings to operate on: please call simulator.rank()"
            )
        elif eval_set not in self.ranks:
            raise RuntimeError(
                f"Simulator has no rankings for eval set {eval_set}: please call simulator.rank()"
            )

        return self.ranks[eval_set]

    def topK(self, k: int, eval_set: str) -> pd.DataFrame:
        """
        Returns the top K ranked strategies from the previous ranking.

        Parameters:
        -----------
        k: int
            The number of strategies to return.
        eval_set:
            For which eval_set to return the top K strategies for.

        Returns:
        --------
        pd.DataFrame:
            The top K ranking strategy configurations for the specified eval_set.
        """
        return self.getRanks(eval_set=eval_set).head(k).copy()

    @staticmethod
    def getStrategies(df: pd.DataFrame) -> dict:
        """
        Generates strategy dictionary containing the strategy configurations for all configurations in df.
        This strategy dictionary is properly formatted and can be directly passed into run() or rank()

        Parameters:
        -----------
        df: pd.DataFrame
            A rank dataframe generated by the rank() function.

        Returns:
        --------
        dict:
            All strategy configurations contained within the dataframe.
        """
        df = df.copy()
        df["params"] = df.apply(lambda x: tuple([*zip(*x["params"].items())]), axis=1)
        df["param_names"] = df.apply(lambda x: x["params"][0], axis=1)
        df["param_values"] = df.apply(lambda x: x["params"][1], axis=1)
        groups = df.groupby(by="strategy")

        strategies = {}
        for strategy, group in groups:
            param_names = group["param_names"].unique()
            param_values = group["param_values"].tolist()

            assert len(param_names) == 1
            strategies[strategy] = param_names[0], param_values

        return strategies

    def addCallback(self, callback: SimulationCallback) -> None:
        """
        Adds a SimulationCallback to this simulator object.

        Parameters:
        -----------
        callback: SimulationCallback
            The callback to add to this simulator object.
        """
        if not isinstance(callback, SimulationCallback):
            raise ValueError(f"Callbacks must be of type {type(SimulationCallback)}!")

        if callback.has_save_artifacts:
            callback.path = self.output_dir

        if not self.callbacks:
            self.callbacks = []

        self.callbacks.append(callback)

    def _runCallbacks(self, method: str, **kwargs):
        if self.callbacks:
            for callback in self.callbacks:
                func = getattr(callback, method)
                func(**kwargs)

    def clear(self) -> None:
        """
        Clears all data, including any simulations run, curves loaded, or ranks generated.
        """
        self.simulations = None
        self.tasks = []
        self.ranks = {}

    def _process_tasks_parallel(self, **kwargs) -> list[pd.DataFrame]:
        if not ray.is_initialized():
            ray.init()

        futures = [
            self._process_task_remote.remote(self, task=task, **kwargs)
            for task in self.tasks
        ]

        results = [ray.get(future) for future in tqdm(futures)]
        return results

    @ray.remote
    def _process_task_remote(self, task: str, **kwargs) -> pd.DataFrame:
        return self._process_task(task=task, **kwargs)

    def _process_tasks_seq(self, **kwargs) -> list[pd.DataFrame]:
        return [self._process_task(task=task, **kwargs) for task in tqdm(self.tasks)]

    def _process_task(
        self,
        task: str,
        strategies: dict,
        filters: dict,
    ) -> pd.DataFrame:
        meta_data, model_data = load_json(task)
        dataset, fold = self._get_dataset_fold(task)
        task_info = [dataset, fold, meta_data["problem_type"]]

        # TODO: ask nick if this would be a better approach
        # consider constructing a df with all the values iterated over
        # and then using df.apply() on each row
        # we can store numpy arrays as a df cell value
        # calling apply on each row to apply simulations?
        # use wrapper fn to do all the strategy and callback setup steps

        self._runCallbacks(
            "before_task",
            curve_data=model_data,
            strategies=strategies,
            filters=filters,
        )

        results = []
        for model, data in model_data.items():
            if filters["models"] and model not in filters["models"]:
                continue

            eval_sets, metrics, curves = data
            val_index = eval_sets.index("val")
            for i, metric in enumerate(metrics):
                if filters["metrics"] and metric not in filters["metrics"]:
                    continue

                for j, eval_set in enumerate(eval_sets):
                    if filters["eval_sets"] and eval_set not in filters["eval_sets"]:
                        continue

                    if val_index == -1:
                        raise ValueError(
                            "Validation Set not Included in Learning Curves!"
                        )

                    stopping_curve = curves[i][val_index]  # always stop based on val
                    eval_curve = curves[i][j]

                    for strategy, (param_names, param_configs) in strategies.items():
                        kwargs = {}
                        if self.factory.get_strategy_class(
                            strategy
                        ).needs_curve_metadata:
                            kwargs["metadata"] = meta_data
                        strategy = self.factory.make_strategy(strategy, **kwargs)

                        for config in param_configs:
                            params = {
                                name: param for name, param in zip(param_names, config)
                            }
                            strategy.update_params(**params)

                            self._runCallbacks(
                                "before_strategy",
                                model=model,
                                metric=metric,
                                eval_set=eval_set,
                                strategy=strategy,
                            )

                            results.append(
                                task_info
                                + [model, metric, eval_set]
                                + [strategy.name, str(params)]
                                + strategy.simulate(
                                    stopping_curve=stopping_curve, eval_curve=eval_curve
                                )
                            )

                            self._runCallbacks(
                                "after_strategy",
                                model=model,
                                metric=metric,
                                eval_set=eval_set,
                                strategy=strategy,
                            )

        self._runCallbacks("after_task")

        columns = [
            "dataset",
            "fold",
            "problem_type",
            "model",
            "metric",
            "eval_set",
            "strategy",
            "params",
            *AbstractStrategy.simulate_columns,
        ]

        return pd.DataFrame(results, columns=columns)

    def _rank(self, eval_set: str = "val", by: str = "error_then_iter"):
        ranks = self.simulations.copy()
        ranks = ranks[ranks["eval_set"] == eval_set]

        sortby = {
            "error": self._rank_by_error,
            "iter": self._rank_by_iter,
            "error_then_iter": self._rank_by_error_then_iter,
            "iter_then_error": self._rank_by_iter_then_error,
        }

        if by not in sortby:
            raise ValueError(f"Invalid simulation ranking method by={by} provided.")

        sortby = sortby[by]
        ranks["superscore"] = sortby(ranks)

        groups = ["dataset", "fold", "model", "metric"]
        ranks["rank"] = ranks.groupby(groups)["superscore"].rank()

        # use param dict str for grouping (dict is unhashable)
        ranks["groups"] = ranks.apply(lambda x: (x["strategy"], x["params"]), axis=1)
        ranks = ranks.groupby("groups")["rank"].mean()
        ranks = ranks.sort_values().reset_index()

        import json

        ranks["strategy"] = ranks["groups"].apply(lambda x: x[0])
        ranks["params"] = ranks["groups"].apply(
            lambda x: json.loads(x[1].replace("'", '"'))
        )
        ranks = ranks.drop("groups", axis=1)
        return ranks

    def _rank_by_error(self, df: pd.DataFrame):
        return list(df["percent_error_diff"])

    def _rank_by_iter(self, df: pd.DataFrame):
        return list(df["percent_iter_diff"])

    def _rank_by_error_then_iter(self, df: pd.DataFrame):
        return list(zip(df["percent_error_diff"], df["percent_iter_diff"]))

    def _rank_by_iter_then_error(self, df: pd.DataFrame):
        return list(zip(df["percent_iter_diff"], df["percent_error_diff"]))

    def _validate_and_preprocess_init_kwargs(
        self, search_method: str, callbacks: list[Callable], output_dir: str, seed: int
    ):
        # search_method
        valid_search_methods = ("random", "grid")
        if search_method not in valid_search_methods:
            raise ValueError(
                f"search_method parameter must be one of {valid_search_methods}!"
            )

        self.search_method = search_method

        # output_dir
        import time

        if not isinstance(output_dir, str):
            raise ValueError(f"Output directory: {output_dir} must be a string!")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = os.path.join(output_dir, f"{output_dir}_{int(time.time())}")
        os.mkdir(self.output_dir)

        # callbacks
        if callbacks:
            for callback in callbacks:
                self.addCallback(callback)

        # seed
        if not isinstance(seed, int):
            raise ValueError(f"Seed must be a valid integer, not {seed}")
        self.seed = seed

    def _validate_and_preprocess_filter(
        self, filter: str | list[str] | None = None, default: list[str] = []
    ):
        if filter is None:
            filter = default
        elif isinstance(filter, str):
            filter = [filter]
        elif isinstance(filter, list) and all(
            isinstance(value, str) for value in filter
        ):
            pass
        else:
            raise ValueError(f"Invalid filter argument: {filter}")

        return filter

    def _preprocess_strategies(
        self, strategies: dict
    ) -> dict[str, tuple[list[str] | list[tuple[int | float]]]]:
        """
        Properly formats parameter values in strategies dictionary.

        Expected input format:
            strategies = {
                strategy_name: {
                    param: int | float,
                    param: list[int | float],
                    param: tuple[int, int, int], # (start, end, step), equivalent to python range
                    ...
                },
                ...
            }

        Returns:
        --------
        dict[str, tuple[list[str] | list[tuple[int | float]]]]:
            Ensures that each strategy now maps to a tuple: (parameter names, parameter configs)
            where parameter names is a list[str] of all parameter names for that strategy and
            where parameter configs is a list[tuple[int | float]] where each tuple represents a
            different configuration of that strategy. The value of each index within these tuples
            corresponds to the same indexed parameter in parameter names.
        """
        # check for preformatted strategy dict (output of getStrategies())
        preformatted = True
        for strategy, params in strategies.items():
            if not (isinstance(params, tuple) and len(params) == 2):
                preformatted = False
                break

            names, values = params
            if not isinstance(names, tuple) or not isinstance(values, list):
                preformatted = False
                break

            if not all(isinstance(name, str) for name in names) or not all(
                isinstance(value, tuple) for value in values
            ):
                preformatted = False
                break

        if preformatted:
            return strategies

        # check for valid strategy input format
        for strategy, params in strategies.items():
            for param, val in params.items():
                if isinstance(val, (float, int)):
                    params[param] = [val]
                elif (
                    type(val) == tuple
                    and len(val) == 3
                    and all(isinstance(num, (float, int)) for num in val)
                ):
                    start, end, step = val
                    params[param] = np.arange(start, end + step, step).tolist()
                elif type(val) == list and all(
                    isinstance(num, (float, int)) for num in val
                ):
                    # already properly formatted
                    pass
                else:
                    raise ValueError(
                        f"Invalid parameter {param}={val} for strategy {strategy}"
                    )

            param_names = list(params.keys())
            param_vals = list(params.values())

            # Defining the parameter search space

            # sklearn parameter sampler class:
            # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterSampler.html#sklearn.model_selection.ParameterSampler

            # https://www.kaggle.com/code/willkoehrsen/intro-to-model-tuning-grid-and-random-search

            # explore these techniques with ray.tune: https://docs.ray.io/en/latest/tune/key-concepts.html
            # https://docs.ray.io/en/latest/tune/faq.html#which-search-algorithm-scheduler-should-i-choose
            # https://docs.ray.io/en/latest/tune/faq.html#how-do-i-configure-search-spaces

            if self.search_method == "grid":
                param_configs = list(itertools.product(*param_vals))
            elif self.search_method == "random":
                random.seed(self.seed)

                # perform sqrt(num grid search configs) random samples
                num_configs = math.prod([len(arr) for arr in param_vals])
                num_samples = round(math.sqrt(num_configs))

                param_configs = [
                    tuple(random.sample(v, k=1)[0] for v in param_vals)
                    for i in range(num_samples)
                ]

            strategies[strategy] = (param_names, param_configs)
            print(f"{strategy}: {len(param_configs)} parameter configurations")

        return strategies

    def _get_dataset_fold(self, path: str) -> tuple[str, str]:
        parts = path.rstrip("/").split("/")

        if len(parts) < 3:
            raise ValueError(
                f"The path {path} does not contain enough parts to extract the last two components: dataset and fold.\n"
                + "expected path pattern: .../dataset/fold/learning_curves.json"
            )

        # expected path pattern: .../dataset/fold/learning_curves.json
        dataset, fold, _ = parts[-3:]

        return dataset, fold


"""
Considerations:
- Can't just compare metric error's directly (diff magnitudes): https://chatgpt.com/share/04b78338-7faa-48cb-9a88-cc096ba9f387
- Diff models have diff number of total iterations (i.e. gbm has ~9999 iterations while nn_torch has < 50), so want to test diff patience values for each model seperately
- How many max iterations / max epochs should we even set when generating learning curves for testing early stopping / during regular training when max iter/epoch is hit
- When choosing optimal parameter configurations, give more weight to the metrics used as defaults for stopping in AutoGluon (i.e. log_loss, mean_absolute_error)


"which stopping strategy to use":
---------------------------------
for each strategy:

    opt score = min(test curve)

    for each metric:
        best score = simulate strategy
        compare best score to opt score


This is a different problem entirely: "which stopping metric to use":
---------------------------------------------------------------------
count times metric beats others

opt score = min(test curve)

for each metric:
    best score = simulate strategy
    compare best score to opt score
    save best score to metric

best metric = min(best scores) # can't compare across metrics like this
"""
