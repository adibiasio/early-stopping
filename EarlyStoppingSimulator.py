import numpy as np
import pandas as pd
from typing import List
import json, itertools, glob, os

from strategies.StrategyFactory import StrategyFactory
from strategies.AbstractStrategy import AbstractStrategy


class StoppingSimulator:
    def __init__(self):
        self.tasks = []
        self.simulations = None
        self.ranks = {}
        self.factory = StrategyFactory()

        self.cached_strategies = None
        self.default_strategies = {
            "simple_patience": { "patience": (1, 50, 1) },
            "linear_adaptive_patience": { "a": (0.01, 0.51, 0.05), "b": (1, 50, 1) },
        }


    def clear(self):
        self.simulations = None
        self.ranks = {}


    def load_curves(self, path):
        if type(path) is list:
            self.tasks.append(path)
            return

        paths = []
        if os.path.isdir(path):
            paths = glob.glob(os.path.join(path, '**', '*.json'), recursive=True)
        elif os.path.isfile(path) and path.lower().endswith('.json'):
            paths.append(path)

        for file in paths:
            with open(file, "r") as f:
                self.tasks.append(json.load(f))


    def run(self, strategies: dict | None = None):
        if strategies is None:
            strategies = self.default_strategies
        elif strategies == self.cached_strategies:
            return self.simulations
        cached_strategies = strategies

        results = []
        for strategy, params in strategies.items():
            results.extend(self.param_search(strategy, **params))

        strategy_columns = ["strategy", "params"]
        task_columns = ["dataset", "tid", "fold", "framework", "problem_type"]
        curve_columns = ["model", "metric", "eval_set"]
        simulate_columns = ["total_iter", 'best_iter', 'opt_iter', "best_error", "opt_error", "error_diff", "percent_error_diff", "percent_iter_diff"]
        all_columns = strategy_columns + task_columns + curve_columns + simulate_columns
        self.simulations = pd.DataFrame(results, columns=all_columns)

        return self.simulations


    def param_search(self, strategy: str, **strategy_params):
        # define search space for simulation: look into sklearn.GridSearchCV
        # either pass single param config, ranges of param configs (s, e, st), or direct list or combination
        for param, val in strategy_params.items():
            if isinstance(val, (float, int)):
                strategy_params[param] = [val]
            elif type(val) == tuple and len(val) == 3 and all(isinstance(num, (float, int)) for num in val):
                start, end, step = val
                strategy_params[param] = np.arange(start, end + step, step).tolist()
            elif type(val) == list and all(isinstance(num, (float, int)) for num in val):
                pass
            else:
                raise ValueError(f"Invalid strategy parameter value: {param}={val}")

        param_names = list(strategy_params.keys())
        param_vals = list(strategy_params.values())
        param_configs = list(itertools.product(*param_vals))

        results = []
        for config in param_configs:
            params = { name: param for name, param in zip(param_names, config) }
            strategy_config = self.factory.make_strategy(strategy, **params)
            results.extend([[strategy_config.name, str(params)] + row for row in self.eval_config(strategy_config)])

        return results

    # (simple_configs + adaptive_configs) * tasks (6) * models (3) * metrics (5avg = 4m, 6b) * eval_sets (3)

    def eval_config(self, strategy: AbstractStrategy):
        results = []
        for task in self.tasks:
            t = task[0]
            task_metadata = [t["dataset"], t["tid"], t["fold"], t["framework"], t["problem_type"]]
            for model, info in task[1].items():
                metric_names, metric_curves = info
                for i, metric in enumerate(metric_names):
                    for j, eval_set in enumerate(["train", "val", "test"]):
                        if j >= len(metric_curves[i]): break
                        stopping_curve = metric_curves[i][1] # always stop on val
                        eval_curve = metric_curves[i][j]
                        results.append(task_metadata + [model, metric, eval_set] + strategy.simulate(stopping_curve=stopping_curve, eval_curve=eval_curve))

        return results


    def rank(self, by: str = "error_then_iter", strategies: dict | None = None, eval_sets: str | List[str] | None = None):
        if self.simulations is None or strategies != self.cached_strategies:
            self.run(strategies=strategies)

        if eval_sets is None:
            eval_sets = ["val", "test"]
        elif isinstance(eval_sets, str):
            eval_sets = [eval_sets]
        elif isinstance(eval_sets, list) and all(isinstance(eval_set, str) for eval_set in eval_sets):
            pass
        else:
            raise ValueError(f"Invalid eval set argument: {eval_sets}")

        self.ranks = {}
        for eval_set in eval_sets:
            self.ranks[eval_set] = self._rank(by=by, eval_set=eval_set)
        
        return self.ranks


    def _rank(self, by: str = "error_then_iter", eval_set: str = "val"):
        ranks = self.simulations.copy()
        ranks = ranks[ranks["eval_set"] == eval_set]

        sortby = {
            "error": self._rank_by_error,
            "iter": self._rank_by_iter,
            "error_then_iter": self._rank_by_error_then_iter,
            "iter_then_error": self._rank_by_iter_then_error,
        }

        if by not in sortby:
            raise ValueError(f"Invalid simulation ranking method (by) provided.")

        sortby = sortby[by]
        ranks['superscore'] = sortby(ranks)

        ranks["rank"] = ranks.groupby(["dataset", "fold", "model", "metric"])["superscore"].rank()
        strategy_name_mapping = ranks.groupby('params')['strategy'].unique().to_dict()

        ranks = ranks.groupby("params")["rank"].mean()
        ranks = ranks.sort_values().reset_index()

        ranks["strategy"] = ranks["params"].map(strategy_name_mapping).apply(lambda x: x[0])

        return ranks


    def _rank_by_error(self, df: pd.DataFrame):
        return list(df['percent_error_diff'])


    def _rank_by_iter(self, df: pd.DataFrame):
        return list(df['percent_iter_diff'])


    def _rank_by_error_then_iter(self, df: pd.DataFrame):
        return list(zip(df['percent_error_diff'], df['percent_iter_diff']))


    def _rank_by_iter_then_error(self, df: pd.DataFrame):
        return list(zip(df['percent_iter_diff'], df['percent_error_diff']))


    def strategies(self):
        """
        Return names of all strategies contained within self.simulations
        """
        pass



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


