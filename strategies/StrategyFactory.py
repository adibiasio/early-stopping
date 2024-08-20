from typing import Type

from strategies.AbstractStrategy import AbstractStrategy
from strategies.AutoGluonStrategy import AutoGluonStrategy
from strategies.FeaturePatienceStrategy import FeaturePatienceStrategy
from strategies.FixedIterationStrategy import FixedIterationStrategy
from strategies.LinearAdaptivePatienceStrategy import LinearAdaptivePatienceStrategy
from strategies.MinDeltaStrategy import MinDeltaStrategy
from strategies.PolynomialAdaptivePatienceStrategy import (
    PolynomialAdaptivePatienceStrategy,
)
from strategies.SimplePatienceStrategy import SimplePatienceStrategy
from strategies.SlidingWindowStrategy import SlidingWindowStrategy


class StrategyFactory:
    """
    General Usage:
        For guidance on supported strategies and their parameters, call
        the help() method on your StrategyFactory object or class.

    Patience Based Stopping Strategies:
        Patience is defined as the number of iterations the training process
        will continue for without any observed improvements in the model before
        stopping. The StrategyFactory supports constant patience values or
        patience defined as a function of the current iteration.

        All patience strategies support bounds for patience values, denoted by
        (minp, maxp). So for the following functions p(x), the overall patience
        functions are best respresented as the following. See below.

            P(x) = min(maxp, max(minp, p(x)))

        Strategies:
            simple_patience: SimplePatienceStrategy
                p(x) = p

            linear_adaptive_patience: LinearAdaptivePatienceStrategy
                p(x) = ax + b

            polynomial_adaptive_patience: PolynomialAdaptivePatienceStrategy
                p(x) = ax^n + b

            feature_patience: FeaturePatienceStrategy
                p(x) = ax^n + b_num_rounds_train

            autogluon_patience: AutoGluonStrategy
                Feature patience with hardcoded parameter values reflecting
                AutoGluon's default strategy settings:
                    a = 0.3
                    b = 10
                    n = 1
                    minp = 20
                    maxp = 300

        Each of the strategies listed above support variations with sliding window
        and minimum delta. Read more about these stopping strategies in the "Non-Patience
        Based" Stopping Strategies section. These variations follow the strategy naming
        scheme below, assuming the original name is STRATEGY:

            Variations:
                sliding_window_STRATEGY
                STRATEGY_with_min_delta
                sliding_window_STRATEGY_with_min_delta

        To use these variations, simply call make_strategy with the umbrella strategy name
        and relevant parameters for that variation. e.g.
            factory = StrategyFactory()
            factory.make_strategy("simple_patience", min_delta=0.05) ==> simple_patience_with_min_delta

    Non-Patience Based Stopping Strategies:
        The following strategies do not utilize patience when determining to stop.

        Strategies:
            fixed_iteration: FixedIterationStrategy
                Always stop at a predefined iteration, N, regardless of changes in
                model performance.

            min_delta: MinDeltaStrategy
                The minimum delta is defined as the minimum change in the performance
                metric needed to qualify as an improvement

            sliding_window: SlidingWindowStrategy
                Calculate the performance metric value at each iteration as the average
                of the previous N iterations.
    """

    _strategy_class_map = {
        "simple_patience": SimplePatienceStrategy,
        "linear_adaptive_patience": LinearAdaptivePatienceStrategy,
        "polynomial_adaptive_patience": PolynomialAdaptivePatienceStrategy,
        "feature_patience": FeaturePatienceStrategy,
        "autogluon_patience": AutoGluonStrategy,
        "fixed_iteration": FixedIterationStrategy,
        "min_delta": MinDeltaStrategy,
        "sliding_window": SlidingWindowStrategy,
    }

    def make_strategy(self, name: str, **kwargs) -> AbstractStrategy:
        if name not in self._strategy_class_map:
            raise ValueError("Invalid Strategy Name")

        strategy = self._strategy_class_map[name]
        return strategy(**kwargs)

    @classmethod
    def strategy_map(cls) -> dict:
        return cls._strategy_class_map

    @classmethod
    def get_strategy_class(cls, name: str) -> Type[AbstractStrategy]:
        if name not in cls._strategy_class_map:
            raise ValueError("Invalid Strategy Name")

        return cls._strategy_class_map[name]

    @classmethod
    def help(cls) -> None:
        print("Supported Strategies:\n")
        for name, strategy in cls._strategy_class_map.items():
            print(f"\t{name}: {strategy.__name__}")
            print(f"\tValid parameters: {strategy.user_params()}\n")
