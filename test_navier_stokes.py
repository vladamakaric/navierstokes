from navier_stokes import local_boundary_normal
import numpy as np
from numpy.testing import assert_array_equal


def unit(v):
    return v / np.linalg.norm(v)


def matrix(m):
    return np.flip(np.array(m), 0)


def test_boundary_normal():
    assert_array_equal(
        boundary_normal(matrix([[0, 1, 1],
                                [0, 0, 0],
                                [0, 0, 0]])), unit([1, 2])
    )

    assert_array_equal(
        boundary_normal(matrix([[0, 1, 1],
                                [0, 0, 1],
                                [0, 0, 0]])), unit([1, 1])
    )

    assert_array_equal(
        boundary_normal(matrix([[0, 0, 1],
                                [0, 0, 1],
                                [0, 0, 0]])), unit([2, 1])
    )

    assert_array_equal(
        boundary_normal(matrix([[0, 0, 1],
                                [0, 0, 1],
                                [0, 0, 1]])), unit([1, 0])
    )

    assert_array_equal(
        boundary_normal(matrix([[0, 0, 1],
                                [0, 0, 1],
                                [0, 1, 1]])), unit([3, -1])
    )

    assert_array_equal(
        boundary_normal(matrix([[0, 0, 1],
                                [0, 0, 1],
                                [1, 1, 1]])), unit([2, -2])
    )

    assert_array_equal(
        boundary_normal(matrix([[0, 0, 0],
                                [0, 0, 1],
                                [1, 1, 1]])), unit([1, -3])
    )

    assert_array_equal(
        boundary_normal(matrix([[0, 0, 0],
                                [0, 0, 0],
                                [1, 1, 1]])), unit([0, -1])
    )

    assert_array_equal(
        boundary_normal(matrix([[0, 0, 0],
                                [0, 0, 0],
                                [0, 1, 0]])), unit([0, -1])
    )
