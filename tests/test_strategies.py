import pytest
import numpy as np

from strategies.SimplePatienceStrategy import SimplePatienceStrategy
from strategies.LinearAdaptivePatienceStrategy import LinearAdaptivePatienceStrategy


def c(n):
    return [0] * n


# TODO: come up with tests for all strategies (more)


# make test for running all strategies just to ensure it is operational


def simple_patience_tests():
    return [
        [10, 6, 16, (-5, 5, -5, 10, -15)],
        [10, 6, 16, (-5, 5, -10),],
        [10, 6, 16, (-5, 10)],
        [10, 1, 11, (10, -15)],
        [11, 11, 22, (-10, 10, -5)],
        [11, 21, 21, (-5, 5, -10)],
        [11, 21, 26, (-5, 5, -10, 5)],
        [10, 6, 16, [-5, 5, *c(5), -5]],
        [16, 26, 31, [-5, 5, *c(5), -10, 5]],
    ]


def adaptive_patience_tests():
    tests = simple_patience_tests()
    tests = [[0] + test for test in tests]

    tests.extend([
        [0.1, 5, 6, 13, (-5, 5, -5)],
    ])

    return tests

# TODO: results format changed, update this

@pytest.mark.parametrize("patience, expected_best_iter, expected_total_iter, intervals", simple_patience_tests())
def test_simple_patience(patience, expected_best_iter, expected_total_iter, intervals):
    curve = make_curve(*intervals)
    strategy = SimplePatienceStrategy(patience=patience)
    results = strategy.simulate(curve=curve)
    total_iter, best_iter, *_ = results
    
    assert best_iter == expected_best_iter
    assert total_iter == expected_total_iter


@pytest.mark.parametrize("patience, expected_best_iter, expected_total_iter, intervals", simple_patience_tests())
def test_adaptive_supports_simple_patience(patience, expected_best_iter, expected_total_iter, intervals):
    curve = make_curve(*intervals)
    simple = SimplePatienceStrategy(patience=patience)
    simple_results = simple.simulate(curve=curve)

    adaptive = LinearAdaptivePatienceStrategy(a=0, b=patience)
    adaptive_results = adaptive.simulate(curve=curve)

    assert simple_results == adaptive_results


@pytest.mark.parametrize("a, b, expected_best_iter, expected_total_iter, intervals", adaptive_patience_tests())
def test_adaptive_patience(a, b, expected_best_iter, expected_total_iter, intervals):
    curve = make_curve(*intervals)
    strategy = LinearAdaptivePatienceStrategy(a=a, b=b)
    results = strategy.simulate(curve=curve)
    total_iter, best_iter, *_ = results

    assert best_iter == expected_best_iter
    assert total_iter == expected_total_iter



# checking that adaptive with a = 0 defaults to simple patience
# """
# df = simulator.simulations.copy()
# df = df[df["eval_set"] == "val"]
# df

# simple = df[df["strategy"] == "simple_patience"]
# adaptive = df[df["strategy"] == "adaptive_patience"]

# simple["p"] = simple["params"].apply(lambda x: json.loads(x.replace("'", '"'))["patience"])

# adaptive["params"] = adaptive["params"].apply(lambda x: json.loads(x.replace("'", '"')))
# adaptive["a"] = adaptive["params"].apply(lambda x: x["a"])
# adaptive["b"] = adaptive["params"].apply(lambda x: x["b"])

# del simple["params"]
# del adaptive["params"]

# simple = simple[simple["p"] <= 50]
# adaptive = adaptive[adaptive["a"] == 0]

# adaptive["p"] = adaptive["b"]
# del adaptive["a"]
# del adaptive["b"]

# adaptive = adaptive.reset_index()
# simple = simple.reset_index()

# del simple["index"]
# del adaptive["index"]

# del simple["strategy"]
# del adaptive["strategy"]

# from pandas.testing import assert_frame_equal
# assert_frame_equal(simple, adaptive)"""




def make_curve_tests():
    return [
        [-5, 5, -5, 10, -15],
    ]


@pytest.mark.parametrize("intervals", make_curve_tests())
def test_make_curve_util(intervals):
    curve = make_curve(*intervals)

    length = sum([abs(interval) for interval in intervals]) + 1
    assert len(curve) == length

    # check_intervals = []

    # i = 1
    # while i < length:
    #     current_interval = 0

    #     is_inc = (curve[i] - curve[i - 1] / abs(curve[i] - curve[i - 1])) > 0



    #     while curve[i] - curve[i - 1] > 0 if


    # for i in range(2):
    #     pass


    # assert check_intervals == intervals


# what about constant curves? => *c(n)
def make_curve(*intervals):
    """
    make_curve(-40, 30, -20, 10)
    ==> curve will decrease for 40, inc for 30, dec for 20, inc for 10
    ==> total length of curve is sum(params) + 1

    make_curve(0) will keep curve constant for an interval of length 1

    to make constant interval of length n, use helper method c like so:
    make_curve(*c(n)) = [n, n, n, ...] where len(curve) is n + 1

    You can use constant intervals in combination with regular ones:
    make_curve(-5, 5, *c(5), -5)
    """
    start = sum([abs(interval) if interval != 0 else 1 for interval in intervals])
    curve = np.zeros(start + 1)
    curve[0] = start
    
    i = 0
    for interval in intervals:
        if interval == 0:
            curve[i + 1] = curve[i]
            i += 1
            continue
        step = int(interval / abs(interval))
        curve[i+1:i+1+abs(interval)] = list(np.arange(curve[i] + step, curve[i] + interval + step, step))
        i += abs(interval)

    return curve.tolist()

