import numpy as np
import sympy as sp
from numpy.testing import assert_array_equal
import navier_stokes
from pytest import approx


def matrix(m):
    return np.flip(np.array(m), 0)


def test_cell_types():
    f, b, i = "f", "b", "i"

    def cell_type_letter(cell: navier_stokes.Cell):
        match cell:
            case navier_stokes.FluidCell():
                return f
            case navier_stokes.BoundaryCell():
                return b
            case navier_stokes.ObstacleInteriorCell():
                return i
            case _:
                raise ValueError(f"Unexpected type: {type(cell).__name__}")

    for grid, expected_types in [
        (
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [f, f, f, f, f],
                [f, f, b, b, f],
                [f, b, i, b, f],
                [f, b, b, b, f],
                [f, f, f, f, f],
            ],
        ),
        (
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [f, f, f, f, f],
                [f, b, b, b, f],
                [f, b, b, b, f],
                [f, f, f, f, f],
            ],
        ),
        (
            [
                [1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            [
                [b, b, b, b, b, b],
                [f, f, f, f, f, f],
                [f, f, f, f, f, f],
                [f, f, f, f, f, f],
                [b, b, b, b, b, b],
            ],
        ),
        (
            [
                [1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            [
                [b, b, b, b, b, b],
                [f, f, f, f, f, f],
                [f, f, f, f, f, f],
                [f, b, b, b, b, f],
                [f, b, i, i, b, f],
                [f, b, i, i, b, f],
                [f, b, b, b, b, f],
                [f, f, f, f, f, f],
                [f, f, f, f, f, f],
                [b, b, b, b, b, b],
            ],
        ),
        (
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [f, f, f, f, f, f],
                [f, f, f, b, b, f],
                [f, f, f, b, b, f],
                [f, b, b, i, b, f],
                [f, b, i, i, b, f],
                [f, b, b, b, b, f],
                [f, f, f, f, f, f],
            ],
        ),
    ]:
        cells = navier_stokes.cells(matrix(grid))
        cellTypes = np.vectorize(cell_type_letter)(cells)
        assert_array_equal(cellTypes, matrix(expected_types))


def test_normals():
    xx = np.array([0, 0])
    no = np.array([0, 1])
    ea = np.array([1, 0])
    so = np.array([0, -1])
    we = np.array([-1, 0])
    nw = np.array([-1, 1]) / np.sqrt(2)
    ne = np.array([1, 1]) / np.sqrt(2)
    sw = np.array([-1, -1]) / np.sqrt(2)
    se = np.array([1, -1]) / np.sqrt(2)

    for grid, expected_normals in [
        (
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [xx, xx, xx, xx, xx],
                [xx, xx, se, sw, xx],
                [xx, se, xx, we, xx],
                [xx, ne, no, nw, xx],
                [xx, xx, xx, xx, xx],
            ],
        ),
        (
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [xx, xx, xx, xx, xx, xx],
                [xx, xx, xx, se, sw, xx],
                [xx, xx, xx, ea, we, xx],
                [xx, se, so, xx, we, xx],
                [xx, ne, no, no, nw, xx],
                [xx, xx, xx, xx, xx, xx],
            ],
        ),
        (
            [
                [1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            [
                [no, no, no, no, no, no],
                [xx, xx, xx, xx, xx, xx],
                [xx, xx, xx, xx, xx, xx],
                [so, so, so, so, so, so],
            ],
        ),
    ]:
        cells = navier_stokes.cells(matrix(grid))

        def cell_normal(single_cell_arr):
            if isinstance(single_cell_arr[0], navier_stokes.BoundaryCell):
                return single_cell_arr[0].normal
            else:
                return np.array([0.0, 0.0])

        normals = np.apply_along_axis(
            cell_normal, axis=1, arr=cells.reshape(-1, 1)
        ).reshape(cells.shape + (2,))
        assert_array_equal(normals, matrix(expected_normals))


def test_fluid_cell_num():
    grid = matrix(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
    )
    expected_cell_nums = matrix(
        [
            [12, 13, 14, 15, 16],
            [9, 10, -1, -1, 11],
            [7, -1, -1, -1, 8],
            [5, -1, -1, -1, 6],
            [0, 1, 2, 3, 4],
        ],
    )

    def cell_num(cell: navier_stokes.Cell):
        if isinstance(cell, navier_stokes.FluidCell):
            return cell.num
        return -1

    cells = navier_stokes.cells(grid)
    num_matrix = np.vectorize(cell_num)(cells)
    assert_array_equal(num_matrix, expected_cell_nums)


def test_projection_matrix():
    grid = matrix(
        [
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ],
    )

    cells = navier_stokes.cells(grid)
    fluid_cells = [c for c in cells.flat if isinstance(c, navier_stokes.FluidCell)]
    div_free_projection = navier_stokes.DivergenceFreeProjection(cells)
    A = div_free_projection.A
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
    cells = navier_stokes.cells(grid)
    div_free_projection = navier_stokes.DivergenceFreeProjection(cells)
    A = div_free_projection.A
    b = div_free_projection._projection_b(w)
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
    cells = navier_stokes.cells(
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
    helmholtzDecomposition = navier_stokes.DivergenceFreeProjection(cells)
    residuals = []
    helmholtzDecomposition.project(velocity_field, residuals)
    assert residuals[-1] < 1e-4


def test_helmholtz_decomposition_boundary_condtitions():
    cells = navier_stokes.cells(
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
    helmholtzDecomposition = navier_stokes.DivergenceFreeProjection(cells)
    residuals = []
    helmholtzDecomposition.project(velocity_field, residuals)
    assert residuals[-1] < 1e-4
    # No flow in boundary normal direction, in any of the 6 boundary cells.
    assert np.dot([1, 1], velocity_field[2][2]) == approx(0)
    assert np.dot([1, 0], velocity_field[3][2]) == approx(0)
    assert np.dot([1, -1], velocity_field[4][2]) == approx(0)
    assert np.dot([-1, 1], velocity_field[2][3]) == approx(0)
    assert np.dot([-1, 0], velocity_field[3][3]) == approx(0)
    assert np.dot([-1, -1], velocity_field[4][3]) == approx(0)


def test_helmholtz_decomposition_non_zero_divergence():
    cells = navier_stokes.cells(np.zeros(shape=(20, 20)))
    velocity_field = np.zeros(cells.shape + (2,))
    for j, i in np.ndindex(cells.shape):
        if 10 <= j <= 15 and 10 <= i <= 15:
            velocity_field[j][i] = [2, 0]
    helmholtzDecomposition = navier_stokes.DivergenceFreeProjection(cells)
    residuals = []
    helmholtzDecomposition.project(velocity_field, residuals)
    assert residuals[-1] < 1e-4
