from navier_stokes import local_boundary_normal, normal_and_diffs
import numpy as np
from numpy.testing import assert_array_equal


def unit(v):
    return v / np.linalg.norm(v)


def matrix(m):
    return np.flip(np.array(m), 0)

def test_normal_and_diffs():
    normal, x_diff, y_diff = normal_and_diffs(matrix([[0, 1, 1],
                                                      [0, 0, 0],
                                                      [0, 0, 0]]),
                                                      inward_corner_difference=True,
                                                      simple_normals=True)
    assert_array_equal(np.array([0, 1]), normal)
    assert x_diff == (1,-1)
    assert y_diff == (0,-1)

    normal, x_diff, y_diff = normal_and_diffs(matrix([[0, 0, 1],
                                                      [0, 0, 0],
                                                      [0, 0, 0]]),
                                                      inward_corner_difference=True,
                                                      simple_normals=True)
    assert_array_equal(unit([1, 1]), normal)
    assert x_diff == (1,0)
    assert y_diff == (1,0)

    normal, x_diff, y_diff = normal_and_diffs(matrix([[1, 0, 0],
                                                      [0, 0, 0],
                                                      [0, 0, 0]]),
                                                      inward_corner_difference=True,
                                                      simple_normals=True)
    assert_array_equal(unit([-1, 1]), normal)
    assert x_diff == (0,-1)
    assert y_diff == (1,0)

    normal, x_diff, y_diff = normal_and_diffs(matrix([[0, 0, 0],
                                                      [0, 0, 0],
                                                      [0, 0, 1]]),
                                                      inward_corner_difference=True,
                                                      simple_normals=True)
    assert_array_equal(unit([1, -1]), normal)
    assert x_diff == (1,0)
    assert y_diff == (0,-1)

    normal, x_diff, y_diff = normal_and_diffs(matrix([[0, 0, 1],
                                                      [0, 0, 0],
                                                      [0, 0, 0]]),
                                                      inward_corner_difference=False,
                                                      simple_normals=True)
    assert_array_equal(unit([1, 1]), normal)
    assert x_diff == (0,-1)
    assert y_diff == (0,-1)

    normal, x_diff, y_diff = normal_and_diffs(matrix([[0, 0, 0],
                                                      [0, 0, 0],
                                                      [1, 0, 0]]),
                                                      inward_corner_difference=True,
                                                      simple_normals=True)
    assert_array_equal(unit([-1, -1]), normal)
    assert x_diff == (0,-1)
    assert y_diff == (0,-1)


    normal, x_diff, y_diff = normal_and_diffs(matrix([[0, 0, 0],
                                                      [0, 0, 1],
                                                      [0, 1, 1]]),
                                                      inward_corner_difference=True,
                                                      simple_normals=True)
    assert_array_equal(unit([1, -1]), normal)
    assert x_diff == (0,-1)
    assert y_diff == (1,0)

    normal, x_diff, y_diff = normal_and_diffs(matrix([[0, 0, 0],
                                                      [0, 0, 1],
                                                      [0, 0, 0]]),
                                                      inward_corner_difference=True,
                                                      simple_normals=True)
    assert_array_equal(unit([1, 0]), normal)
    assert x_diff == (0,-1)
    assert y_diff == (1,-1)


def test_boundary_normal():
    assert_array_equal(
        local_boundary_normal(matrix([[0, 1, 1], [0, 0, 0], [0, 0, 0]])), unit([1, 2])
    )

    assert_array_equal(
        local_boundary_normal(matrix([[0, 1, 1], [0, 0, 1], [0, 0, 0]])), unit([1, 1])
    )

    assert_array_equal(
        local_boundary_normal(matrix([[0, 0, 1], [0, 0, 1], [0, 0, 0]])), unit([2, 1])
    )

    assert_array_equal(
        local_boundary_normal(matrix([[0, 0, 1], [0, 0, 1], [0, 0, 1]])), unit([1, 0])
    )

    assert_array_equal(
        local_boundary_normal(matrix([[0, 0, 1], [0, 0, 1], [0, 1, 1]])), unit([3, -1])
    )

    assert_array_equal(
        local_boundary_normal(matrix([[0, 0, 1], [0, 0, 1], [1, 1, 1]])), unit([2, -2])
    )

    assert_array_equal(
        local_boundary_normal(matrix([[0, 0, 0], [0, 0, 1], [1, 1, 1]])), unit([1, -3])
    )

    assert_array_equal(
        local_boundary_normal(matrix([[0, 0, 0], [0, 0, 0], [1, 1, 1]])), unit([0, -1])
    )

    assert_array_equal(
        local_boundary_normal(matrix([[0, 0, 0], [0, 0, 0], [0, 1, 0]])), unit([0, -1])
    )
