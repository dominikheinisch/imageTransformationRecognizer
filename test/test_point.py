import pytest

from point import Point


def test_eq():
    point_1 = Point([0, 0])
    point_2 = Point([0, 0.0])
    assert point_1 == point_2
    point_1 = Point([0, 1])
    point_2 = Point([0, 1.000001])
    assert point_1 == point_2
    point_1 = Point([0, 1])
    point_2 = Point([0, 1.0000011])
    assert not point_1 == point_2
    point_1 = Point([1000000, 1])
    point_2 = Point([1000001, 1])
    assert point_1 == point_2
    point_1 = Point([1000000, 1])
    point_2 = Point([1000001.0001, 1])
    assert not point_1 == point_2