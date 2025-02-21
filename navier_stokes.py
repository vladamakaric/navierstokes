import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import pyamg
from numpy.typing import NDArray


class Simulator:
    def __init__(self, grid: NDArray[np.int_], viscosity: float):
        self.cells = cells(grid)
        self.viscosity = viscosity
        self.div_free_projection = DivergenceFreeProjection(self.cells)
        self.velocity = np.zeros(grid.shape + (2,))
        self.indices = CellIndices(self.cells)

    def advect(self, dt: float):
        def sample_velocity(pos):
            return bilinear_interpolate(self.velocity, pos.reshape(-1, 2)).reshape(
                pos.shape
            )

        mid_pos = self.indices.positions - self.velocity * dt / 2
        mid_v = sample_velocity(mid_pos)
        new_pos = self.indices.positions - mid_v * dt
        self.velocity = sample_velocity(new_pos)

    def diffuse(self, dt: float):
        s = self.indices.stencil
        w = self.velocity
        laplacian = (
            -4 * w[s.j, s.i]
            + w[s.j_up, s.i]
            + w[s.j_down, s.i]
            + w[s.j, s.i_right]
            + w[s.j, s.i_left]
        )
        # Don't want to diffuse the obstacle cells.
        laplacian[self.indices.obstacle] = [0, 0]
        self.velocity += self.viscosity * laplacian * dt

    def step(self, dt: float, force: float):
        self.velocity[self.indices.non_obstacle] += np.array([force, 0]) * dt
        self.advect(dt)
        self.diffuse(dt)
        self.div_free_projection.project(self.velocity)


@dataclass(frozen=True)
class Cell:
    j: int
    i: int
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


def cells(grid):
    height, width = grid.shape
    cells = np.empty(shape=grid.shape, dtype=Cell)
    fluid_cell_count = 0
    for index in np.argwhere(grid == 0):
        j, i = index[0], index[1]
        cells[tuple(index)] = FluidCell(j, i, cells=cells, num=fluid_cell_count)
        fluid_cell_count += 1
    for index in np.argwhere(grid == 1):
        j, i = index[0], index[1]
        fluid_dirs = [
            (jd, id)
            for (jd, id) in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if not grid[(j + jd) % height][(i + id) % width]
        ]
        if len(fluid_dirs) == 0:
            cells[j][i] = ObstacleInteriorCell(j, i, cells=cells)
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
            j, i, cells, normal / np.linalg.norm(normal), x_difference, y_difference
        )
    return cells


class DivergenceFreeProjection:
    """Projects the velocity vector field to its divergence-free part.

    This is applied at the end of each simulation step, to ensure the
    velocity field satisfies mass continuuity, ie. incompressibility.

    Implementation:

    Every vector field w is the sum of a divergence free part u, and the
    gradient of a scalar field p: w = u + grad(p); aka Helmholtz Decomposition.

    This class finds p, and subtracts its gradient from w. p is determined
    by the Poisson equation, which we get by taking the divergence of the
    Helmhotlz Decomposition: div(w) = div(grad(p)) = laplacian(p).

    We set the boundary conditions of p such that the resulting div-free
    field (w - grapd(p)) flows around the boundary, ie. there is no flow in
    the direction normal to the boundary:

    normal*(w-grad(p)) = 0    =>    grad(p) = normal*w

    This is a so-called Neumann boundary condition.
    """

    def __init__(self, cells):
        self.cells = cells
        self.indices = CellIndices(cells)
        self.A = DivergenceFreeProjection._coefficient_matrix(self.cells[self.indices.fluid])
        self.multigrid_solver = pyamg.ruge_stuben_solver(self.A)
        self.boundary_gradient_stencil = (
            DivergenceFreeProjection._boundary_gradient_stencil(cells)
        )
        self.boundary_normals = np.zeros(cells.shape + (2,))
        for cell in cells.flat:
            match cell:
                case BoundaryCell(index=index, normal=normal):
                    self.boundary_normals[index] = normal / np.sum(np.abs(normal))

    def project(self, w: NDArray[np.float_]):
        b = self._projection_b(w)
        # Can extract residuals from this solve method for debugging.
        x = self.multigrid_solver.solve(b)
        P = np.zeros(shape=self.cells.shape)
        P[self.indices.fluid] = x
        for c in self.cells.flat:
            match c:
                case BoundaryCell(index=index, normal=normal, x_diff=xd, y_diff=yd):
                    v = w[index]
                    # g = (other_x*abs(nx) + other_y*abs(ny) + b)/(np.abs(nx + ny))
                    an_x = np.abs(normal[0])
                    an_y = np.abs(normal[1])
                    P[index] = np.dot(v, normal) / (an_x + an_y)
                    if xd:
                        P[index] += P[xd.fluid_cell.index] * an_x / (an_x + an_y)
                    if yd:
                        P[index] += P[yd.fluid_cell.index] * an_y / (an_x + an_y)
        bs = self.boundary_gradient_stencil
        w[:, :, 0] -= (P[bs.j, bs.i_right] - P[bs.j, bs.i_left]) / bs.id
        w[:, :, 1] -= (P[bs.j_up, bs.i] - P[bs.j_down, bs.i]) / bs.jd
        w[self.indices.obstacle] = [0, 0]

    def _coefficient_matrix(fluid_cells):
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

    def _projection_b(self, w):
        s = standard_central_diff_stencil(self.cells.shape)
        wdotn = np.sum(w * self.boundary_normals, axis=-1)
        divergence = (
            w[s.j, s.i_right, 0]
            - w[s.j, s.i_left, 0]
            + w[s.j_up, s.i, 1]
            - w[s.j_down, s.i, 1]
        ) / 2
        boundary_condition_rhs = (
            wdotn[s.j, s.i_right]
            + wdotn[s.j, s.i_left]
            + wdotn[s.j_up, s.i]
            + wdotn[s.j_down, s.i]
        )
        return (
            divergence[self.indices.fluid] - boundary_condition_rhs[self.indices.fluid]
        )

    def _boundary_gradient_stencil(cells):
        """Modify central diff stencil to single diff at boundaries."""
        s = standard_central_diff_stencil(cells.shape)
        id = np.full(cells.shape, 2)
        jd = np.full(cells.shape, 2)
        for cell in cells.flat:
            if not isinstance(cell, BoundaryCell):
                continue
            if cell.x_diff:
                if isinstance(cell.left, FluidCell):
                    s.i_right[cell.index] = cell.i
                else:
                    s.i_left[cell.index] = cell.i
                id[cell.index] = 1
            if cell.y_diff:
                if isinstance(cell.down, FluidCell):
                    s.j_up[cell.index] = cell.j
                else:
                    s.j_down[cell.index] = cell.j
                jd[cell.index] = 1
            # Handle concave corners, where a flat boundary is next to an inside obstacle cell,
            # like so (center is the horizontal, flat, boundary):
            #
            #    1,1,1
            #    1,1,1
            #    0,0,1
            #
            if not cell.x_diff:
                if isinstance(cell.left, ObstacleInteriorCell):
                    s.i_left[cell.index] = cell.i
                    id[cell.index] = 1
                    print(cell.index)
                if isinstance(cell.right, ObstacleInteriorCell):
                    s.i_right[cell.index] = cell.i
                    id[cell.index] = 1
                    print(cell.index)
            if not cell.y_diff:
                if isinstance(cell.up, ObstacleInteriorCell):
                    s.j_up[cell.index] = cell.j
                    jd[cell.index] = 1
                    print(cell.index)
                if isinstance(cell.down, ObstacleInteriorCell):
                    s.j_down[cell.index] = cell.j
                    jd[cell.index] = 1
                    print(cell.index)

        return FiniteDifferenceStencil(
            j=s.j,
            i=s.i,
            j_up=s.j_up,
            j_down=s.j_down,
            i_right=s.i_right,
            i_left=s.i_left,
            jd=jd,
            id=id,
        )


def standard_central_diff_stencil(shape):
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


def bilinear_interpolate(m: NDArray, p: NDArray):
    """Interpolate between the values of grid m, for each position in array p."""

    def broadcast_multiply(a, b):
        a_expanded = a.reshape(a.shape + (1,) * (b.ndim - 1))
        return a_expanded * b

    ll = np.floor(p).astype(int)
    ur = ll + [1, 1]

    # Corners
    Ia = m[ll[:, 1] % m.shape[0], ll[:, 0] % m.shape[1]]
    Ib = m[ur[:, 1] % m.shape[0], ll[:, 0] % m.shape[1]]
    Ic = m[ll[:, 1] % m.shape[0], ur[:, 0] % m.shape[1]]
    Id = m[ur[:, 1] % m.shape[0], ur[:, 0] % m.shape[1]]

    # Corner weights
    wa = (ur[:, 0] - p[:, 0]) * (ur[:, 1] - p[:, 1])
    wb = (ur[:, 0] - p[:, 0]) * (p[:, 1] - ll[:, 1])
    wc = (p[:, 0] - ll[:, 0]) * (ur[:, 1] - p[:, 1])
    wd = (p[:, 0] - ll[:, 0]) * (p[:, 1] - ll[:, 1])
    return (
        broadcast_multiply(wa, Ia)
        + broadcast_multiply(wb, Ib)
        + broadcast_multiply(wc, Ic)
        + broadcast_multiply(wd, Id)
    )


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


# @dataclass(frozen=True)
# TODO: Figure out how to make this frozen.
class CellIndices:
    """Various cell indices for vectorized operations."""

    positions: NDArray
    stencil: FiniteDifferenceStencil
    obstacle: Tuple[NDArray, NDArray]
    non_obstacle: Tuple[NDArray, NDArray]
    fluid: Tuple[NDArray, NDArray]

    def __init__(self, cells):
        xs, ys = np.meshgrid(np.arange(cells.shape[1]), np.arange(cells.shape[0]))
        self.positions = np.stack([xs, ys], axis=2, dtype=np.float64)
        self.fluid = ([], [])
        self.obstacle = ([], [])
        self.non_obstacle = ([], [])
        self.stencil = standard_central_diff_stencil(cells.shape)
        for cell in cells.flat:
            match cell:
                case FluidCell():
                    self.fluid[0].append(cell.j)
                    self.fluid[1].append(cell.i)
                    self.non_obstacle[0].append(cell.j)
                    self.non_obstacle[1].append(cell.i)
                case BoundaryCell():
                    self.non_obstacle[0].append(cell.j)
                    self.non_obstacle[1].append(cell.i)
                case ObstacleInteriorCell():
                    self.obstacle[0].append(cell.j)
                    self.obstacle[1].append(cell.i)