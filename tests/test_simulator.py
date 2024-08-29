"""
Ensure that all strategies and possible configurations
run without errors (note this does not check for strategy correctness).
"""

import pytest
import itertools

from early_stopping.EarlyStoppingSimulator import StoppingSimulator
from early_stopping.strategies.StrategyFactory import StrategyFactory

# create tests that check that running a simulation with a model and strategy config yields exactly the same result 
# when running with these configs in autogluon (when training)


def strategy_configs():
    configs = []
    strategies = StrategyFactory.strategy_map()
    for strategy, strategy_class in strategies.items():

        params = strategy_class.user_params()

        all_combinations = list(itertools.chain.from_iterable(
            itertools.combinations(params, r) for r in range(1, len(params) + 1)
        ))

        for config in all_combinations:
            configs.append({ strategy : { param : 1 for param in config } })

    return configs


# To see test parameterization names, add ids=str to decorator, but custom ids not supported in vscode test ui
@pytest.mark.parametrize("config", strategy_configs())
def test_all_strategies(config):
    simulator = StoppingSimulator()
    simulator.load_curves("curves")
    simulator.rank(strategies=config)


@pytest.fixture()
def simulator():
    simulator = StoppingSimulator()
    simulator.load_curves("learning_curves/2/")
    return simulator


