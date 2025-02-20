import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
import pyamg
import sympy
from numpy.typing import NDArray


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


def fluid_cell_matrix_to_array_index(cells):
    index = ([], [])
    for cell in cells.flat:
        if isinstance(cell, FluidCell):
            index[0].append(cell.j)
            index[1].append(cell.i)
    return index


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


def bilinear_interpolate(m, p):
    # Needed to make interpolation work for fields of vectors, and scalars.
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


def boundary_normal_field(cells):
    boundary_normal_field = np.zeros(cells.shape + (2,))

    for cell in cells.flat:
        match cell:
            case BoundaryCell(index=index, normal=normal):
                boundary_normal_field[index] = normal / np.sum(np.abs(normal))

    return boundary_normal_field


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


def divergence_of_velocity_field(vf, s: FiniteDifferenceStencil):
    return (
        vf[s.j, s.i_right, 0]
        - vf[s.j, s.i_left, 0]
        + vf[s.j_up, s.i, 1]
        - vf[s.j_down, s.i, 1]
    ) / 2


def projection_b(w, index, normals, s: FiniteDifferenceStencil):
    wdotn = np.sum(w * normals, axis=-1)
    b = (
        divergence_of_velocity_field(w, s)[index]
        - (
            wdotn[s.j, s.i_right]
            + wdotn[s.j, s.i_left]
            + wdotn[s.j_up, s.i]
            + wdotn[s.j_down, s.i]
        )[index]
    )
    return b


def standard_central_diff_stencil(shape):
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


def boundary_gradient_stencil(cells):
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


class HelmholtzDecomposition:
    def __init__(self, cells):
        self.stencil = standard_central_diff_stencil(cells.shape)
        self.boundary_gradient_stencil = boundary_gradient_stencil(cells)
        self.obstacle_cell_index = ([], [])
        for cell in cells.flat:
            if isinstance(cell, ObstacleInteriorCell):
                self.obstacle_cell_index[0].append(cell.j)
                self.obstacle_cell_index[1].append(cell.i)

        self.cells = cells
        self.fluid_cells = [c for c in self.cells.flat if isinstance(c, FluidCell)]
        self.fluid_cell_matrix_to_array_index = fluid_cell_matrix_to_array_index(
            self.cells
        )
        self.boundary_normals = boundary_normal_field(self.cells)
        self.A = projection_A(self.fluid_cells)
        self.multigrid_solver = pyamg.ruge_stuben_solver(self.A)
        self.P = np.zeros(shape=self.cells.shape)

    def gradientField(self, w, residuals=None):
        b = projection_b(
            w,
            self.fluid_cell_matrix_to_array_index,
            self.boundary_normals,
            self.stencil,
        )
        x = self.multigrid_solver.solve(b, tol=1e-5, maxiter=100, residuals=residuals)
        for c in self.fluid_cells:
            self.P[c.index] = x[c.num]
        for c in self.cells.flat:
            match c:
                case BoundaryCell(index=index, normal=normal, x_diff=xd, y_diff=yd):
                    v = w[index]
                    # g = (other_x*abs(nx) + other_y*abs(ny) + b)/(np.abs(nx + ny))
                    an_x = np.abs(normal[0])
                    an_y = np.abs(normal[1])
                    self.P[index] = np.dot(v, normal) / (an_x + an_y)
                    if xd:
                        self.P[index] += (
                            self.P[xd.fluid_cell.index] * an_x / (an_x + an_y)
                        )
                    if yd:
                        self.P[index] += (
                            self.P[yd.fluid_cell.index] * an_y / (an_x + an_y)
                        )
        bs = self.boundary_gradient_stencil
        w[:, :, 0] -= (self.P[bs.j, bs.i_right] - self.P[bs.j, bs.i_left]) / bs.id
        w[:, :, 1] -= (self.P[bs.j_up, bs.i] - self.P[bs.j_down, bs.i]) / bs.jd
        w[self.obstacle_cell_index] = [0, 0]
        return self.P


class Simulator:
    def __init__(self, grid):
        self.cells = cells(grid)
        xs, ys = np.meshgrid(np.arange(grid.shape[1]), np.arange(grid.shape[0]))
        self.positions = np.stack([xs, ys], axis=2, dtype=np.float64)
        self.wc = np.zeros(grid.shape + (2,))
        self.velocity_field = np.zeros(grid.shape + (2,))
        # TODO: Both of these are shared between HD and Simulator. Refactor.
        self.stencil = standard_central_diff_stencil(self.cells.shape)
        self.obstacle_cell_index = ([], [])
        self.non_obstacle_cell_index = ([], [])
        for cell in self.cells.flat:
            if isinstance(cell, ObstacleInteriorCell):
                self.obstacle_cell_index[0].append(cell.j)
                self.obstacle_cell_index[1].append(cell.i)
            else:
                self.non_obstacle_cell_index[0].append(cell.j)
                self.non_obstacle_cell_index[1].append(cell.i)
        self.helmholtz_decomposition = HelmholtzDecomposition(self.cells)

    def advect(self, dt):
        # RK2
        mid_pos = self.positions - self.velocity_field * dt / 2
        mid_v = bilinear_interpolate(
            self.velocity_field, mid_pos.reshape(-1, 2)
        ).reshape(mid_pos.shape)
        new_pos = self.positions - mid_v * dt
        self.velocity_field = bilinear_interpolate(
            self.velocity_field, new_pos.reshape(-1, 2)
        ).reshape(new_pos.shape)

    def diffuse(self, dt):
        viscosity_constant = 11
        s = self.stencil
        w = self.velocity_field
        laplacian = (
            -4 * w[s.j, s.i]
            + w[s.j_up, s.i]
            + w[s.j_down, s.i]
            + w[s.j, s.i_right]
            + w[s.j, s.i_left]
        )
        # Don't want to diffuse the obstacle cells.
        laplacian[self.obstacle_cell_index] = [0, 0]
        self.velocity_field += viscosity_constant * laplacian * dt

    def step(self, dt, force, projection_residuals=None):
        self.velocity_field[self.non_obstacle_cell_index] += np.array([force, 0]) * dt
        self.advect(dt)
        self.diffuse(dt)
        P = self.helmholtz_decomposition.gradientField(
            self.velocity_field, projection_residuals
        )
        return P
