import numpy as np
from numpy.testing import assert_array_equal
import cell


def matrix(m):
    # Flip the y axis so that it increases in the up direciton.
    return np.flip(np.array(m), 0)


def test_cell_types():
    f, b, i = "f", "b", "i"

    def cell_type_letter(c: cell.Cell):
        match c:
            case cell.Fluid():
                return f
            case cell.Boundary():
                return b
            case cell.ObstacleInterior():
                return i
            case _:
                raise ValueError(f"Unexpected type: {type(c).__name__}")

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
        cells = cell.create_cell_grid(matrix(grid))
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
        cells = cell.create_cell_grid(matrix(grid))

        def cell_normal(single_cell_arr):
            if isinstance(single_cell_arr[0], cell.Boundary):
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

    def cell_num(c: cell.Cell):
        if isinstance(c, cell.Fluid):
            return c.num
        return -1

    cells = cell.create_cell_grid(grid)
    num_matrix = np.vectorize(cell_num)(cells)
    assert_array_equal(num_matrix, expected_cell_nums)
