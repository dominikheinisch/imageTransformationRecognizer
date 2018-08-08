import numpy as np

import ransac
from point import Point


def test_is_invertible():
    m = np.array([[0, 1]])
    assert ransac.is_invertible(m) == False
    m = np.array([[0, 0],
                  [0, 1]])
    assert ransac.is_invertible(m) == False
    m = np.array([[1]])
    assert ransac.is_invertible(m) == True
    m = np.array([[1, 0],
                  [0, 1]])
    assert ransac.is_invertible(m) == True


def test_calc_model():
    model = np.array([
        [Point([0, 0]), Point([1, 1])],
        [Point([1, 1]), Point([2, 2])],
        [Point([0, 1]), Point([1, 2])]
    ])
    result = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
    ])
    assert np.array_equal(ransac.calc_model(model), result) == True
