import pytest

from ransac import *
from point import Point


def test_calculate_ransac_pairs():
    filtered_pairs = [(Point([1, 1]), Point([1, 1])), (Point([1, 1]), Point([2, 2]))]
    model = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
    ])
    max_error = 1
    assert calculate_ransac_pairs(filtered_pairs, model, max_error) == [(Point([1, 1]), Point([2, 2]))]
    filtered_pairs = [(Point([1, 1]), Point([1, 1])), (Point([1, 1]), Point([2, 2]))]
    model = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    max_error = 1
    assert calculate_ransac_pairs(filtered_pairs, model, max_error) == [(Point([1, 1]), Point([1.0, 1.0]))]

def test_model_error():
    model = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    pair = (Point([1, 1]), Point([1, 1]))
    assert model_error(model, pair) == 0
    model = np.array([
        [1, 0, 2],
        [0, 1.5, 0],
        [0, 0, 1],
    ])
    pair = (Point([1, 1]), Point([4, 1.5]))
    assert 1 - 1e8 < abs(model_error(model, pair) - 0) < 1 + 1e8


def test_calc_model():
    samples = np.array([
        [Point([0, 0]), Point([0, 0])],
        [Point([1, 1]), Point([1, 1])],
        [Point([0, 1]), Point([0, 1])]
    ])
    result = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    assert np.array_equal(calc_model(samples), result) == True
    samples = np.array([
        [Point([0, 0]), Point([0, 0])],
        [Point([1, 1]), Point([1, 1])]
    ])
    with pytest.raises(IndexError):
        calc_model(samples)


def test_affine_array():
    points = [(Point([0, 0]), Point([0, 0])), (Point([0, 1]), Point([0, 1])), (Point([0, 2]), Point([0, 2]))]
    assert affine_array(points) == None
    points = [(Point([0, 0]), Point([0, 0])), (Point([1, 1]), Point([1, 1])), (Point([0, 1]), Point([0, 1]))]
    result = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    assert np.array_equal(affine_array(points), result) == True
    points = [(Point([0, 0]), Point([2, 3])), (Point([1, 1]), Point([3, 4])), (Point([0, 1]), Point([2, 4]))]
    result = np.array([
        [1, 0, 2],
        [0, 1, 3],
        [0, 0, 1],
    ])
    assert np.array_equal(affine_array(points), result) == True
    points = [(Point([0, 0]), Point([0, 0])), (Point([1, 1]), Point([3, 1])), (Point([0, 1]), Point([0, 1]))]
    result = np.array([
        [3, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    assert np.array_equal(affine_array(points), result) == True
    points = [(Point([0, 0]), Point([0, 0])), (Point([1, 1]), Point([1, 1.5])), (Point([0, 1]), Point([0, 1.5]))]
    result = np.array([
        [1, 0, 0],
        [0, 1.5, 0],
        [0, 0, 1],
    ])
    assert np.allclose(affine_array(points), result, atol=1e-010) == True


def test_perspective_array():
    # TODO
    pass


def test_is_invertible():
    m = np.array([[]])
    assert is_invertible(m) == False
    m = np.array([[0, 1]])
    assert is_invertible(m) == False
    m = np.array([[0, 0],
                  [0, 1]])
    assert is_invertible(m) == False
    m = np.array([[1]])
    assert is_invertible(m) == True
    m = np.array([[1, 0],
                  [0, 1]])
    assert is_invertible(m) == True


def test_get_params():
    assert get_params([(Point([0, 5]), Point([2, 3])), (Point([10, 14]), Point([12, 13]))]) \
           == (0, 5, 2, 3, 10, 14, 12, 13)
    assert get_params([]) == ()
