import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Literal
from numpy.typing import NDArray


@dataclass
class Cell:
    j: int
    i: int

@dataclass
class FluidCell(Cell):
    num: int

@dataclass
class BoundaryCell(Cell):
    @dataclass
    class Difference:
        # (fluid_cell - self)*dir
        fluid_cell: FluidCell
        dir: Literal[-1, 1]
    # Points from the fluid into the boundary.
    normal: NDArray
    x_difference: Optional[Difference]
    y_difference: Optional[Difference]

@dataclass
class ObstacleInteriorCell(Cell):
    pass


def fluid_cell_index(fluid_cells, grid_shape):
    index = np.empty(shape=grid_shape, dtype=Cell)
    for fluid_cell in fluid_cells:
        index[fluid_cell.grid_index] = fluid_cell
    return index


def projection_A(fluid_cells, fluid_cell_index, grid_shape):
    # step = 1
    # divW = 0
    height, width = grid_shape
    non_boundary_cells = [cell for cell in fluid_cells if cell.boundary_normal is None]
    w = np.array([1, 0])
    b = np.zeros(len(non_boundary_cells))
    A = np.zeros(shape=(len(non_boundary_cells), len(non_boundary_cells)))
    for fluid_cell in non_boundary_cells:
        # One linear equation per non-boundary fluid cell.
        row = A[fluid_cell.non_boundary_index]
        j, i = fluid_cell.grid_index
        row[fluid_cell.non_boundary_index] = -4

        # Hacky div calculation for [1,0] w vector.
        right = fluid_cell_index[j][(i + 1) % width]
        left = fluid_cell_index[j][(i - 1) % width]
        divW = (1 if right.boundary_normal is None else 0) - (
            1 if left.boundary_normal is None else 0
        )
        b[fluid_cell.non_boundary_index] += divW

        for j_offset, i_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            cj = (j + j_offset) % height
            ci = (i + i_offset) % width
            cell = fluid_cell_index[cj][ci]
            if cell.boundary_normal is None:
                row[cell.non_boundary_index] = 1
            else:
                normal = cell.boundary_normal
                x_dir = -int(np.sign(normal[0]))
                y_dir = -int(np.sign(normal[1]))
                an_x = np.abs(normal[0])
                an_y = np.abs(normal[1])
                # This formula is correct:
                # (g-other_x)*abs(nx) + (g-other_y)*abs(ny) = R
                # g = (other_x*abs(nx) + other_y*abs(ny) + b)/(np.abs(nx + ny))
                print((cj, ci))
                if x_dir:
                    other_x = fluid_cell_index[
                        cj, (ci + x_dir) % width
                    ].non_boundary_index
                    row[other_x] += an_x / (an_x + an_y)
                if y_dir:
                    other_y = fluid_cell_index[
                        (cj + y_dir) % height, ci
                    ].non_boundary_index
                    row[other_y] += an_y / (an_x + an_y)
                b[fluid_cell.non_boundary_index] += -np.dot(w, normal)
    return A, b


def fluid_cells(grid):
    height, width = grid.shape
    cells = [] # TODO: Make a grid of Cells
    fluid_cell_count = 0
    # TODO: do a first pass to identify all the Fluid cells, need it for the boundary cells.
    for j, i in np.ndindex(grid.shape):
        if not grid[j][i]:
            cells += [FluidCell(j,i, num=fluid_cell_count)]
            fluid_cell_count += 1
            continue
        empty_neighbors = [
            (jd, id)
            for (jd, id) in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if not grid[(j + jd) % height][(i + id) % width]
        ]
        if len(empty_neighbors) == 0:
            cells += [ObstacleInteriorCell(j,i)]
            continue
        # Boundary
        assert len(empty_neighbors) < 3
        normal = -np.sum(
            [np.flip(np.array(en)) for en in empty_neighbors], axis=0
        )
        assert normal.any()  # [0,0] means one point thick boundary

        cells += [BoundaryCell(j,i, normal / np.linalg.norm(normal))]
    return cells
