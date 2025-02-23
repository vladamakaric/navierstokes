"""Types representing cells of the discrete fluid grid."""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
from numpy.typing import NDArray


@dataclass(frozen=True)
class Cell:
    j: int
    i: int
    # Entire cell grid.
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

    @property
    def index(self):
        return (self.j, self.i)


@dataclass(frozen=True)
class Fluid(Cell):
    # Serial number for mapping all fluid cells on the 2D grid into an array.
    num: int


@dataclass(frozen=True)
class Boundary(Cell):
    @dataclass(frozen=True)
    class Difference:
        # (fluid_cell - self)*dir
        fluid_cell: Fluid
        dir: Literal[-1, 1]

    # Unit vector pointing into the obstacle (away from the fluid).
    normal: np.ndarray
    # Contains the neighboring fluid cells in the x and y directions for single
    # difference approximations of the derivative at the boundary.
    x_diff: Optional[Difference]
    y_diff: Optional[Difference]


@dataclass(frozen=True)
class ObstacleInterior(Cell):
    pass


def create_cell_grid(grid):
    """Creates a grid of cells from a binary grid.

    0 represents fluid, 1 represents the obstacle, but 1s at the boundary
    (who neighbor fluid cells), are special fluid cells called boundary cells.
    Fluid flows in the boundary cells, but it must be tangent to the boundary,
    this is why each boundary cell has a normal.
    """
    height, width = grid.shape
    cells = np.empty(shape=grid.shape, dtype=Cell)
    fluid_cell_count = 0
    for index in np.argwhere(grid == 0):
        j, i = index[0], index[1]
        cells[tuple(index)] = Fluid(j, i, cells=cells, num=fluid_cell_count)
        fluid_cell_count += 1
    for index in np.argwhere(grid == 1):
        j, i = index[0], index[1]
        fluid_dirs = [
            (jd, id)
            for (jd, id) in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if not grid[(j + jd) % height][(i + id) % width]
        ]
        if len(fluid_dirs) == 0:
            cells[j][i] = ObstacleInterior(j, i, cells=cells)
            continue
        fluid_x_dir = [id for jd, id in fluid_dirs if jd == 0]
        fluid_y_dir = [jd for jd, id in fluid_dirs if id == 0]
        assert len(fluid_x_dir) < 2 and len(fluid_y_dir) < 2
        x_difference = None
        y_difference = None
        normal = np.array([0, 0])
        if fluid_x_dir:
            normal[0] = -fluid_x_dir[0]
            x_difference = Boundary.Difference(
                cells[j][(i + fluid_x_dir[0]) % width], fluid_x_dir[0]
            )
        if fluid_y_dir:
            normal[1] = -fluid_y_dir[0]
            y_difference = Boundary.Difference(
                cells[(j + fluid_y_dir[0]) % height][i], fluid_y_dir[0]
            )
        cells[j][i] = Boundary(
            j, i, cells, normal / np.linalg.norm(normal), x_difference, y_difference
        )
    return cells


@dataclass(frozen=True)
class FiniteDifferenceStencil:
    j: NDArray[np.int_]
    i: NDArray[np.int_]
    j_up: NDArray[np.int_]
    j_down: NDArray[np.int_]
    i_left: NDArray[np.int_]
    i_right: NDArray[np.int_]
    id: np.float_ | NDArray[np.float_]
    jd: np.float_ | NDArray[np.float_]


@dataclass(frozen=True)
class CellIndices:
    """Various cell indices for vectorized operations."""

    positions: NDArray
    stencil: FiniteDifferenceStencil
    obstacle: Tuple[NDArray, NDArray]
    non_obstacle: Tuple[NDArray, NDArray]
    fluid: Tuple[NDArray, NDArray]


def create_indices(cells) -> CellIndices:
    xs, ys = np.meshgrid(np.arange(cells.shape[1]), np.arange(cells.shape[0]))
    positions = np.stack([xs, ys], axis=2, dtype=np.float64)
    fluid = ([], [])
    obstacle = ([], [])
    non_obstacle = ([], [])
    for cell in cells.flat:
        match cell:
            case Fluid():
                fluid[0].append(cell.j)
                fluid[1].append(cell.i)
                non_obstacle[0].append(cell.j)
                non_obstacle[1].append(cell.i)
            case Boundary():
                non_obstacle[0].append(cell.j)
                non_obstacle[1].append(cell.i)
            case ObstacleInterior():
                obstacle[0].append(cell.j)
                obstacle[1].append(cell.i)
    return CellIndices(
        positions=positions,
        stencil=central_difference_stencil(cells.shape),
        fluid=fluid,
        obstacle=obstacle,
        non_obstacle=non_obstacle,
    )


def central_difference_stencil(shape):
    """Central difference stencil for gradients divergences and laplacians."""
    j, i = np.indices(dimensions=shape)
    j_up = (j + 1) % shape[0]
    j_down = (j - 1) % shape[0]
    i_right = (i + 1) % shape[1]
    i_left = (i - 1) % shape[1]
    id = 2
    jd = 2
    return FiniteDifferenceStencil(
        j=j, i=i, j_up=j_up, j_down=j_down, i_right=i_right, i_left=i_left, jd=jd, id=id
    )
