from callbacks import (
    GraphSimulationCallback,
    LearningCurveStrategyCallback,
    PatienceStrategyCallback,
)
from EarlyStoppingSimulator import StoppingSimulator

##### Step 1: Define callbacks #####

callbacks = None

# define any callbacks to be used for visualizing simulations
# callbacks = [
#     GraphSimulationCallback(strategy_callback=PatienceStrategyCallback),
#     GraphSimulationCallback(strategy_callback=LearningCurveStrategyCallback),
# ]


##### Step 2: Create the stopping simulator with desired settings #####
simulator = StoppingSimulator(
    search_method="random",  # or grid
    callbacks=callbacks,
    verbosity=4,
)


##### Step 3: Load your learning curves #####

# point to local directory
path = "curves/3" # limit to /adult/0 for much faster runs

# point to s3 bucket
# path = "s3://andrew-bench-dev/aggregated/tabular/ag_bench_learning_curves_full_20240815T102317/learning_curves/"

# if pointing to s3, files will be downloaded to output run folder by default
# provide alternative location here:
save_path = None
# save_path = "curves/my_new_curves"

simulator.load_curves(path, save_path=save_path)


##### Step 4: Rank #####

# uncomment this line to see what strategies are available and their associated parameters
# print(simulator.factory.help())

# sample parameter value ranges
a = {"a": (0, 0.5, 0.05)}
b = {"b": (0, 50, 5)}
p = {"patience": (0, 50, 5)}
d = {"degree": (0.05, 1.3, 0.05)}
sw = {"sliding_window": (1, 7, 1)}
md = {"min_delta": (0, 0.01, 0.001)}
minp = {"min_patience": (0, 100, 10)}
maxp = {"max_patience": (200, 600, 10)}

# use "search_method" field to specify for a strategy a different search method than the global
simulator.rank(
    mode="ray",  # control mode, set to "seq" to run sequentially (enables debugging)
    # filter learning curves by...
    eval_sets=["val", "test"],
    # metrics=["accuracy"],
    # models=["LightGBM"],
    strategies={
        "autogluon_patience": {"search_method": "grid", "simple": [0, 1]},
        "simple_patience": p,
        "linear_adaptive_patience": a | b,
        "polynomial_adaptive_patience": a | b | d,  # | maxp | minp
        "feature_patience": a | b | d,  # | maxp | minp
        # add more (see help() output)
    },
)


# TODO: need to work out some issues with strategy classes
# ##### Step 5: Post-Ranking Functionality #####

# # get ranks df for an eval_set
# val_ranks = simulator.getRanks("val")

# # get top k ranked strategies
# k = 25
# topK = simulator.topK(k, "val")

# # get strategy dict from a rank df
# kStrats = simulator.getStrategies(topK)

# # add additional callbacks and visualize results of
# # smaller number of strategies
# simulator.addCallback(
#     GraphSimulationCallback(strategy_callback=PatienceStrategyCallback)
# )

# simulator.rank(mode="seq", strategies=kStrats)
