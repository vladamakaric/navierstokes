import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal

# TODO: Find a fast rendering canvas, once I have it I can do advection, first in the
# most trivial way, and then maybe do Runge Kutta 4 or whatever. Diffusion I can also do
# in a super simple way first, and then if it blows up I can do the fancy implicit linear
# eq. solver (seems like good experinece to linearize one more thing).

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
    cells: np.ndarray

    @property
    def up(self):
        return self.cells[(self.j + 1) % self.cells.shape[0]][self.i]

    @property
    def down(self):
        return self.cells[(self.j - 1) % self.cells.shape[0]][self.i]

    @property
    def right(self):
        return self.cells[self.j][(self.i + 1) % self.cells.shape[1]]

    @property
    def left(self):
        return self.cells[self.j][(self.i - 1) % self.cells.shape[1]]

    @property
    def neighbors(self):
        return [self.up, self.right, self.down, self.left]


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


def projection_A(fluid_cells):
    A = np.zeros(shape=(len(fluid_cells), len(fluid_cells)))
    for fluid_cell in fluid_cells:
        # One linear equation per fluid cell.
        row = A[fluid_cell.num]
        row[fluid_cell.num] = -4
        for neighbor in fluid_cell.neighbors:
            match neighbor:
                case FluidCell(num=num):
                    row[num] = 1
                case BoundaryCell(normal=normal, x_diff=x_diff, y_diff=y_diff):
                    an_x = np.abs(normal[0])
                    an_y = np.abs(normal[1])
                    # Value of this ghost cell (g) obeys the following boundary condition:
                    # (g-other_x)*abs(nx) + (g-other_y)*abs(ny) = n*w
                    # g = (other_x*abs(nx) + other_y*abs(ny) + n*w)/(np.abs(nx + ny))
                    if x_diff:
                        row[x_diff.fluid_cell.num] += an_x / (an_x + an_y)
                    if y_diff:
                        row[y_diff.fluid_cell.num] += an_y / (an_x + an_y)
                case _:
                    raise ValueError("Neighbor must be a fluid or boundary cell")
    return A


def projection_b(fluid_cells, w):
    b = np.zeros(len(fluid_cells))
    for fc in fluid_cells:
        xdiff = (w[fc.right.index][0] - w[fc.left.index][0]) / 2
        ydiff = (w[fc.up.index][1] - w[fc.down.index][1]) / 2
        divergence = xdiff + ydiff
        b[fc.num] = divergence
        for neighbor in fc.neighbors:
            match neighbor:
                case BoundaryCell(normal=normal):
                    b[fc.num] += -np.dot(w[neighbor.index], normal) / np.sum(
                        np.abs(normal)
                    )
    return b


def cells(grid):
    height, width = grid.shape
    cells = np.empty(shape=grid.shape, dtype=Cell)
    fluid_cell_count = 0
    for index in np.argwhere(grid == 0):
        j, i = index[0], index[1]
        cells[tuple(index)] = FluidCell(j, i, num=fluid_cell_count, cells=cells)
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
