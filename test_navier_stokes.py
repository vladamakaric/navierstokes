import numpy as np
from numpy.testing import assert_array_equal

def unit(v):
    return v / np.linalg.norm(v)


def matrix(m):
    return np.flip(np.array(m), 0)

# def test_normal_and_diffs():
#     normal, x_diff, y_diff = normal_and_diffs(matrix([[0, 1, 1],
#                                                       [0, 0, 0],
#                                                       [0, 0, 0]]),
#                                                       inward_corner_difference=True,
#                                                       simple_normals=True)
#     assert_array_equal(np.array([0, 1]), normal)
#     assert x_diff == (1,-1)
#     assert y_diff == (0,-1)