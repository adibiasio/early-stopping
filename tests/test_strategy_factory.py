import pytest

from early_stopping.strategies.FixedIterationStrategy import FixedIterationStrategy
from early_stopping.strategies.LinearAdaptivePatienceStrategy import LinearAdaptivePatienceStrategy
from early_stopping.strategies.MinDeltaStrategy import MinDeltaStrategy
from early_stopping.strategies.PolynomialAdaptivePatienceStrategy import (
    PolynomialAdaptivePatienceStrategy,
)
from early_stopping.strategies.SimplePatienceStrategy import SimplePatienceStrategy
from early_stopping.strategies.SlidingWindowStrategy import SlidingWindowStrategy
from early_stopping.strategies.StrategyFactory import StrategyFactory


def strategy_classes():
    return [
        ("simple_patience", SimplePatienceStrategy),
        ("linear_adaptive_patience", LinearAdaptivePatienceStrategy),
        ("polynomial_adaptive_patience", PolynomialAdaptivePatienceStrategy),
        ("fixed_iteration", FixedIterationStrategy),
        ("sliding_window", SlidingWindowStrategy),
        ("min_delta", MinDeltaStrategy),
    ]


@pytest.mark.parametrize("strategy, strategy_type", strategy_classes())
def test_strategy_types(strategy, strategy_type, factory):
    strategy = factory.make_strategy(strategy)
    assert isinstance(strategy, strategy_type)
    assert strategy.name == strategy


def patience_classes():
    return [
        "simple_patience",
        "linear_adaptive_patience",
        "polynomial_adaptive_patience",
    ]


@pytest.mark.parametrize("strategy", patience_classes())
def test_patience_variations(strategy, factory):
    base = factory.make_strategy(strategy).name

    min_delta = {"min_delta": 0.05}
    name = factory.make_strategy(strategy, **min_delta).name
    assert name == f"{base}_with_min_delta"

    sliding_window = {"sliding_window": 5}
    name = factory.make_strategy(strategy, **sliding_window).name
    assert name == f"sliding_window_{base}"

    sliding_window_and_min_delta = {**min_delta, **sliding_window}
    name = factory.make_strategy(strategy, **sliding_window_and_min_delta).name
    assert name == f"sliding_window_{base}_with_min_delta"


def test_min_delta_variations(factory):
    sliding_window = {"sliding_window": 5}
    name = factory.make_strategy("min_delta", **sliding_window).name
    assert name == f"sliding_window_min_delta"


def test_sliding_window_variations(factory):
    min_delta = {"min_delta": 0.05}
    name = factory.make_strategy("sliding_window", **min_delta).name
    assert name == f"sliding_window_with_min_delta"


@pytest.fixture()
def factory():
    return StrategyFactory()
