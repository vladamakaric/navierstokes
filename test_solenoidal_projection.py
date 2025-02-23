import numpy as np
import sympy as sp
from numpy.testing import assert_array_equal
import cell
import solenoidal_projection
from pytest import approx


def matrix(m):
    # Flip the y axis so that it increases in the up direciton.
    return np.flip(np.array(m), 0)


def test_coefficient_matrix():
    grid = matrix(
        [
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ],
    )

    cells = cell.create_cell_grid(grid)
    projection = solenoidal_projection.SolenoidalProjection(cells)
    A = projection.A
    assert cells[2][1].num == 4
    assert_array_equal(A[4], np.array([0, 1, 0, 1, -3, 1]))

    grid = matrix(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
        ],
    )

    w = np.full(grid.shape + (2,), [1, 0])
    cells = cell.create_cell_grid(grid)
    projection = solenoidal_projection.SolenoidalProjection(cells)
    A = projection.A
    b = projection._right_hand_side_vector(w)
    assert cells[1][1].num == 5
    n_x = 1 / np.sqrt(2)
    n_y = 1 / np.sqrt(2)
    # Boundary condition for the num=6 (c_6) ghost cell:
    #   (c_6 - c_5)*n_x + (c_6 - c_2)*n_y = w*n
    # We use this condition to express c_6 in terms of other fluid cells:
    #   c_6 = c_5*n_x / (n_x + n_y) + (c_2*n_y + w*n) / (n_x + n_y)
    # That is then substituted in the laplacian linear equation for c_5 below:
    expected_equation_array = np.array(
        [0, 1, n_y / (n_x + n_y), 0, 1, -4 + n_x / (n_x + n_y), 0, 1, 0, 0, 0, 0]
    )
    # w*n / (n_x + n_y) goes to the RHS of the equation (b).
    expected_rhs = -np.dot([1, 0], [n_x, n_y]) / (n_x + n_y)
    assert_array_equal(A[5], expected_equation_array)
    assert b[5] == expected_rhs


def test_helmholtz_decomposition_boundary_and_interior_constraints():
    cells = cell.create_cell_grid(
        matrix(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )
    velocity_field = np.zeros(cells.shape + (2,))
    velocity_field[3:6, 2:4] = [2, 0]
    projection = solenoidal_projection.SolenoidalProjection(cells)
    residuals = []
    projection.project(velocity_field, residuals)
    assert residuals[-1] < 1e-4


def test_helmholtz_decomposition_boundary_condtitions():
    cells = cell.create_cell_grid(
        matrix(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
    )
    velocity_field = np.full(cells.shape + (2,), [2.0, 0])
    for j, i in np.ndindex((cells.shape[0], cells.shape[1])):
        if 2 <= j <= 4 and 2 <= i <= 3:
            velocity_field[j][i] = [0, 0]
    projection = solenoidal_projection.SolenoidalProjection(cells)
    residuals = []
    projection.project(velocity_field, residuals)
    assert residuals[-1] < 1e-4
    # No flow in boundary normal direction, in any of the 6 boundary cells.
    assert np.dot([1, 1], velocity_field[2][2]) == approx(0)
    assert np.dot([1, 0], velocity_field[3][2]) == approx(0)
    assert np.dot([1, -1], velocity_field[4][2]) == approx(0)
    assert np.dot([-1, 1], velocity_field[2][3]) == approx(0)
    assert np.dot([-1, 0], velocity_field[3][3]) == approx(0)
    assert np.dot([-1, -1], velocity_field[4][3]) == approx(0)


def test_helmholtz_decomposition_non_zero_divergence():
    cells = cell.create_cell_grid(np.zeros(shape=(20, 20)))
    velocity_field = np.zeros(cells.shape + (2,))
    for j, i in np.ndindex(cells.shape):
        if 10 <= j <= 15 and 10 <= i <= 15:
            velocity_field[j][i] = [2, 0]

    projection = solenoidal_projection.SolenoidalProjection(cells)
    residuals = []
    projection.project(velocity_field, residuals)
    assert residuals[-1] < 1e-4
