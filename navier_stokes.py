import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass(frozen=True)
class Cell:
    j: int
    i: int

    @property
    def index(self):
        return (self.j, self.i)


@dataclass(frozen=True)
class FluidCell(Cell):
    num: int


@dataclass(frozen=True)
class BoundaryCell(Cell):
    @dataclass(frozen=True)
    class Difference:
        # (fluid_cell - self)*dir
        fluid_cell: FluidCell
        dir: Literal[-1, 1]

    # Points from the fluid into the boundary.
    normal: np.ndarray
    x_diff: Optional[Difference]
    y_diff: Optional[Difference]


@dataclass(frozen=True)
class ObstacleInteriorCell(Cell):
    pass


def projection_A(cells):
    # step = 1
    # divW = 0
    height, width = cells.shape
    fluid_cells = [cell for cell in cells.flat if isinstance(cell, FluidCell)]
    w = np.array([1, 0])
    b = np.zeros(len(fluid_cells))
    A = np.zeros(shape=(len(fluid_cells), len(fluid_cells)))
    for fluid_cell in fluid_cells:
        # One linear equation per non-boundary fluid cell.
        row = A[fluid_cell.num]
        j, i = fluid_cell.j, fluid_cell.i
        row[fluid_cell.num] = -4

        # Hacky div calculation for [1,0] w vector.
        right = cells[j][(i + 1) % width]
        left = cells[j][(i - 1) % width]
        divW = (
            (0 if isinstance(right, BoundaryCell) else 1)
            - (0 if isinstance(left, BoundaryCell) else 1)
        ) / 2
        b[fluid_cell.num] += divW

        # TODO: Consider just replacing all this offset stuff with left() right() functions on the cell object.
        for j_offset, i_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            cj = (j + j_offset) % height
            ci = (i + i_offset) % width
            match cells[cj][ci]:
                case FluidCell(num=num):
                    row[num] = 1
                case BoundaryCell(normal=normal, x_diff=x_diff, y_diff=y_diff):
                    an_x = np.abs(normal[0])
                    an_y = np.abs(normal[1])
                    # This formula is correct:
                    # (g-other_x)*abs(nx) + (g-other_y)*abs(ny) = R
                    # g = (other_x*abs(nx) + other_y*abs(ny) + b)/(np.abs(nx + ny))
                    if x_diff:
                        row[x_diff.fluid_cell.num] += an_x / (an_x + an_y)
                    if y_diff:
                        row[y_diff.fluid_cell.num] += an_y / (an_x + an_y)
                    b[fluid_cell.num] += -np.dot(w, normal) / (an_x + an_y)
                case _:
                    raise ValueError("Neighbor must be a fluid or boundary cell")
    return A, b


def cells(grid):
    height, width = grid.shape
    cells = np.empty(shape=grid.shape, dtype=Cell)
    fluid_cell_count = 0
    for index in np.argwhere(grid == 0):
        j, i = index[0], index[1]
        cells[tuple(index)] = FluidCell(j, i, num=fluid_cell_count)
        fluid_cell_count += 1
    for index in np.argwhere(grid == 1):
        j, i = index[0], index[1]
        fluid_dirs = [
            (jd, id)
            for (jd, id) in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if not grid[(j + jd) % height][(i + id) % width]
        ]
        if len(fluid_dirs) == 0:
            cells[j][i] = ObstacleInteriorCell(j, i)
            continue
        fluid_x_dir = [id for jd, id in fluid_dirs if jd == 0]
        fluid_y_dir = [jd for jd, id in fluid_dirs if id == 0]
        assert len(fluid_x_dir) < 2 and len(fluid_y_dir) < 2
        x_difference = None
        y_difference = None
        normal = np.array([0, 0])
        if fluid_x_dir:
            normal[0] = -fluid_x_dir[0]
            x_difference = BoundaryCell.Difference(
                cells[j][(i + fluid_x_dir[0]) % width], fluid_x_dir[0]
            )
        if fluid_y_dir:
            normal[1] = -fluid_y_dir[0]
            y_difference = BoundaryCell.Difference(
                cells[(j + fluid_y_dir[0]) % height][i], fluid_y_dir[0]
            )
        cells[j][i] = BoundaryCell(
            j, i, normal / np.linalg.norm(normal), x_difference, y_difference
        )
    return cells
