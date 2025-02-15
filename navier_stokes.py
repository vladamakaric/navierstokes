import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
import pyamg
import sympy

# TODO: Find a fast rendering canvas, once I have it I can do advection, first in the
# most trivial way, and then maybe do Runge Kutta 4 or whatever. Diffusion I can also do
# in a super simple way first, and then if it blows up I can do the fancy implicit linear
# eq. solver (seems like good experinece to linearize one more thing).


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


def linearInterpolation(x, xa, xb, ya, yb):
    assert xa <= x <= xb
    if xa == xb:
        assert np.array_equal(ya, yb)
        return ya
    c = (x - xa) / (xb - xa)
    return ya + c * (yb - ya)


def bilinearInterpolation(x, y, xa, xb, ya, yb, zaa, zab, zbb, zba):
    """
    zaa ---- zba
     |        |
     |  zxy   |
     |        |
    zaa ---- zba
    """
    assert (xa <= x <= xb) and (ya <= y <= yb)
    zia = linearInterpolation(x, xa, xb, zaa, zba)
    zib = linearInterpolation(x, xa, xb, zab, zbb)
    return linearInterpolation(y, ya, yb, zia, zib)


def sampleVelocityField(pos, vf):
    # TODO: test.
    i_left = int(np.floor(pos[0]))
    i_right = int(np.ceil(pos[0]))
    j_down = int(np.floor(pos[1]))
    j_up = int(np.ceil(pos[1]))
    height = vf.shape[0]
    width = vf.shape[1]
    return bilinearInterpolation(
        x=pos[0],
        y=pos[1],
        xa=i_left,
        xb=i_right,
        ya=j_down,
        yb=j_up,
        zaa=vf[j_down % height][i_left % width],
        zab=vf[j_up % height][i_left % width],
        zba=vf[j_down % height][i_right % width],
        zbb=vf[j_up % height][i_right % width],
    )


def rungeKutta2(pos, dt, vf):
    # TODO: test.
    v = sampleVelocityField(pos, vf)
    pos_mid = pos + v * dt / 2
    v_mid = sampleVelocityField(pos_mid, vf)
    return pos + v_mid * dt


def streamline(pos, dt, steps, vf):
    points = [pos]
    latest = np.copy(pos)
    for _ in range(steps):
        v = sampleVelocityField(latest, vf)
        latest += v * dt / steps
        points += [np.copy(latest)]
    return points


def trace(pos, dt, steps, velocity_field, dir=1, savePath=False):
    path = []
    currPoint = np.copy(pos)
    for _ in range(steps):
        v = dir * sampleVelocityField(currPoint, velocity_field)
        currPoint += v * dt / steps
        # TODO: Handle the case of tracing into an obstacle. At least sound an alarm.
        if savePath:
            path += [np.copy(currPoint)]
    return currPoint, path


def advect(velocity_field, dt):
    advected_field = np.copy(velocity_field)
    for index in np.ndindex((velocity_field.shape[0], velocity_field.shape[1])):
        pos = np.array([index[1], index[0]], dtype=np.float64)
        endpoint, _ = trace(
            pos=pos, dt=dt, steps=5, velocity_field=velocity_field, dir=-1
        )
        advected_field[index] = sampleVelocityField(endpoint, velocity_field)
    return advected_field


def gradientStencil(cell):
    # Fluid cell default.
    right = cell.right
    left = cell.left
    up = cell.up
    down = cell.down
    dx, dy = 2, 2
    if isinstance(cell, BoundaryCell):
        # sharp corners of boundary cells are forbidden, like this one:

        # 0,0,1,1
        # 0,1,1,0
        # 1,1,0,0
        # 1,0,0,0

        # A boundary cell cannot have 1 hor and 1 ver neighbor (ie. sharp corner).
        # The only way a boundary cell can have two neighbors is if it's a vertical
        # or horizontal edge.

        # 0,0,0,0,0
        # 0,1,1,1,0
        # 1,1,1,1,1
        # 1,1,1,1,1

        # Ban this case too, where left is a boundary, and right is an obstacle.
        # 0,0,1
        # 1,1,1
        # 1,1,1
        if isinstance(cell.right, BoundaryCell) or isinstance(cell.left, BoundaryCell):
            dx, dy = 2, 1
            up = cell.up if isinstance(cell.up, FluidCell) else cell
            down = cell.down if isinstance(cell.down, FluidCell) else cell
        elif isinstance(cell.up, BoundaryCell) or isinstance(cell.down, BoundaryCell):
            dx, dy = 1, 2
            right = cell.right if isinstance(cell.right, FluidCell) else cell
            left = cell.left if isinstance(cell.left, FluidCell) else cell
        else:  # corner with two obstacles to the sides.
            dx, dy = 1, 1
            right = cell.right if isinstance(cell.right, FluidCell) else cell
            left = cell.left if isinstance(cell.left, FluidCell) else cell
            up = cell.up if isinstance(cell.up, FluidCell) else cell
            down = cell.down if isinstance(cell.down, FluidCell) else cell
    if dx == 1:
        assert right.left == left
    if dy == 1:
        assert up.down == down
    return right, left, up, down, dx, dy


def expressBoundaryCellInTermsOfFluidCells(bcell: BoundaryCell):
    b = sympy.IndexedBase("b")
    eq = boundaryCellEquation(bcell)
    other_bcell = None
    for term, _ in eq.lhs.as_coefficients_dict().items():
        if term.has(b) and term.indices != bcell.index:
            # Other bcell must be a neighbor.
            other_bcell = next(n for n in bcell.neighbors if n.index == term.indices)
            break
    if not other_bcell:
        (solution,) = sympy.linsolve([eq], b[bcell.index])
        return solution[0]
    eq2 = boundaryCellEquation(other_bcell)
    (solution,) = sympy.linsolve([eq, eq2], (b[bcell.index], b[other_bcell.index]))
    return solution[0]


def boundaryCellEquation(cell: BoundaryCell):
    b, f, w = sympy.symbols("b f w", cls=sympy.IndexedBase)

    def v(cell: Cell):
        return f[cell.index] if isinstance(cell, FluidCell) else b[cell.index]

    nx, ny = cell.normal[0], cell.normal[1]
    right, left, up, down, dx, dy = gradientStencil(cell)
    gradP_dot_n = nx * (v(right) - v(left)) / dx + ny * (v(up) - v(down)) / dy
    w_dot_n = w[cell.index + (1,)] * ny + w[cell.index + (0,)] * nx
    return sympy.Eq(gradP_dot_n, w_dot_n)


def fluidCellEquations(fluid_cells):
    # One linear equation per fluid cell.
    f, w = sympy.symbols("f w", cls=sympy.IndexedBase)
    equations = []
    for fc in fluid_cells:
        divergence = (
            w[fc.right.index + (0,)]
            - w[fc.left.index + (0,)]
            + w[fc.up.index + (1,)]
            - w[fc.down.index + (1,)]
        ) / 2
        laplacian = -4 * f[fc.index]
        for neighbor in fc.neighbors:
            match neighbor:
                case FluidCell():
                    laplacian += f[neighbor.index]
                case BoundaryCell():
                    laplacian += expressBoundaryCellInTermsOfFluidCells(neighbor)
                    pass
                case _:
                    raise ValueError("Neighbor must be a fluid or boundary cell")
        lhs = sympy.Add(*[t for t in laplacian.as_ordered_terms() if not t.has(f)])
        rhs = sympy.Add(*[t for t in laplacian.as_ordered_terms() if t.has(f)])
        equations += [sympy.Eq(lhs, rhs + divergence)]
    return equations


class HelmholtzDecomposition:
    def __init__(self, cells):
        self.cells = cells
        self.fluid_cells = [c for c in self.cells.flat if isinstance(c, FluidCell)]
        self.A = projection_A(self.fluid_cells)
        self.multigrid_solver = pyamg.ruge_stuben_solver(self.A)

    def solenoidalPart(self, velocity_field, residuals=None):
        b = projection_b(self.fluid_cells, velocity_field)
        x = self.multigrid_solver.solve(b, tol=1e-6, maxiter=100, residuals=residuals)
        P = np.zeros(shape=self.cells.shape)
        for c in self.fluid_cells:
            P[c.index] = x[c.num]
        for c in self.cells.flat:
            match c:
                case BoundaryCell(index=index, normal=normal, x_diff=xd, y_diff=yd):
                    v = velocity_field[index]
                    # g = (other_x*abs(nx) + other_y*abs(ny) + b)/(np.abs(nx + ny))
                    an_x = np.abs(normal[0])
                    an_y = np.abs(normal[1])
                    P[index] = np.dot(v, normal) / (an_x + an_y)
                    if xd:
                        P[index] += P[xd.fluid_cell.index] * an_x / (an_x + an_y)
                    if yd:
                        P[index] += P[yd.fluid_cell.index] * an_y / (an_x + an_y)
        gradP = np.zeros(shape=P.shape + (2,))
        for cell in self.cells.flat:
            match cell:
                case FluidCell():
                    gradP[cell.index] = [
                        (P[cell.right.index] - P[cell.left.index]) / 2,
                        (P[cell.up.index] - P[cell.down.index]) / 2,
                    ]
                case BoundaryCell(index=index, x_diff=xd, y_diff=yd):
                    if xd:
                        gradP[index][0] = (P[xd.fluid_cell.index] - P[index]) * xd.dir
                    if yd:
                        gradP[index][1] = (P[yd.fluid_cell.index] - P[index]) * yd.dir
        # TODO: If copying this vector field takes a while, just do this in place.
        return velocity_field - gradP, P


class Simulator:
    def __init__(self, grid):
        self.cells = cells(grid)
        # TODO: Separate out the Helmholtz projection stuff into a separate class
        # and test it.
        self.velocity_field = np.zeros(grid.shape + (2,))
        self.helmholtz_decomposition = HelmholtzDecomposition(self.cells)

    def step(self, dt, force_field, projection_residuals=None):
        self.velocity_field += force_field * dt
        advected_field = advect(self.velocity_field, dt)
        self.velocity_field, _ = self.helmholtz_decomposition.solenoidalPart(
            advected_field, residuals=projection_residuals
        )
        return self.velocity_field


def read_matrix(filename):
    with open(filename) as f:
        return np.flip(
            np.array([list(map(int, line.strip().split(","))) for line in f]), 0
        )
