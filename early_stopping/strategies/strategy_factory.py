from typing import Type

from .abstract_strategy import AbstractStrategy
from .autogluon_patience import AutoGluonStrategy
from .feature_patience import FeaturePatienceStrategy
from .fixed_iterations import FixedIterationStrategy
from .linear_patience import LinearPatienceStrategy
from .min_delta import MinDeltaStrategy
from .polynomial_patience import PolynomialPatienceStrategy
from .simple_patience import SimplePatienceStrategy
from .sliding_window import SlidingWindowStrategy


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
                p(x) = b

            linear_patience: LinearPatienceStrategy
                p(x) = ax + b

            polynomial_patience: PolynomialPatienceStrategy
                p(x) = ax^n + b

            feature_patience: FeaturePatienceStrategy
                p(x) = ax^n + b_num_rounds_train

            autogluon_patience: AutoGluonStrategy
                Feature patience with hardcoded parameter values reflecting
                AutoGluon's default strategy settings:
                    a = 0.3
                    b = 20
                    degree = 1
                    max_offset = 300
                    max_patience = 10000

        Each of the strategies listed above support variations with sliding window
        and minimum delta. Read more about these stopping strategies in the "Non-Patience
        Based" Stopping Strategies section. 

        To use these variations, simply call make_strategy with your supported strategy name
        and relevant parameters for that variation. e.g.
            factory = StrategyFactory()
            factory.make_strategy("simple_patience", min_delta=0.05) ==> simple_patience with min_delta

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
        "linear_patience": LinearPatienceStrategy,
        "polynomial_patience": PolynomialPatienceStrategy,
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
