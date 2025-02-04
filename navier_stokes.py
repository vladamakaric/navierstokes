import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from numpy.typing import NDArray


@dataclass
class FluidCell:
    # Index in the 1D array of fluid cells.
    index: int
    # Index in the 2D grid, which also contains obstacle cells.
    grid_index: Tuple[int, int]
    # Cells on the boundary have a unit normal pointing into the boundary.
    # TODO: If more boundary-specific fields are needed, create a subtype BoundaryCell.
    non_boundary_index: int
    boundary_normal: Optional[NDArray] = None


def fluid_cell_index(fluid_cells, grid_shape):
    index = np.empty(shape=grid_shape, dtype=FluidCell)
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
    cells = []
    count = 0
    non_boundary_cell_count = 0
    for j, i in np.ndindex(grid.shape):
        boundary_normal = None
        if grid[j][i]:
            empty_neighbors = [
                (jd, id)
                for (jd, id) in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                if not grid[(j + jd) % height][(i + id) % width]
            ]
            assert len(empty_neighbors) < 3
            if len(empty_neighbors) == 0:
                continue
            boundary_normal = -np.sum(
                [np.flip(np.array(en)) for en in empty_neighbors], axis=0
            )
            print(boundary_normal)
            assert boundary_normal.any()  # [0,0] means one point thick boundary
            boundary_normal = boundary_normal / np.linalg.norm(boundary_normal)
        cells += [
            FluidCell(
                index=count,
                grid_index=(j, i),
                non_boundary_index=non_boundary_cell_count,
                boundary_normal=boundary_normal,
            )
        ]
        if boundary_normal is None:
            non_boundary_cell_count += 1
        count += 1
    return cells
