
import cell
import pyamg
import numpy as np
from numpy.typing import NDArray


class SolenoidalProjection:
    """Projects the velocity vector field to its solenoidal part.

    The solenoidal part has zero divergence (aka the divergence-free part),
    which is what we need to satisfy the Navier Stokes continuity equation.
    This is applied at the end of each simulation step to make sure the same
    amount of fluid is flowing in and out of every point.

    Implementation:

    Every vector field w is the sum of a solenoidal part u, and the
    gradient of a scalar field p: w = u + grad(p); aka Helmholtz Decomposition.

    This class finds p, and subtracts its gradient from w. p is determined
    by the Poisson equation, which we get by taking the divergence of the
    Helmhotlz Decomposition: div(w) = div(grad(p)) = laplacian(p).

    We set the boundary conditions of p such that the resulting solenoidal 
    field (w - grapd(p)) flows around the boundary, ie. there is no flow in
    the direction normal to the boundary:

    normal*(w-grad(p)) = 0    =>    grad(p) = normal*w

    This is a so-called Neumann boundary condition.
    """

    def __init__(self, cells):
        self.cells = cells
        self.indices = cell.create_indices(cells)
        self.A = SolenoidalProjection._coefficient_matrix(self.cells[self.indices.fluid])
        self.multigrid_solver = pyamg.ruge_stuben_solver(self.A)
        self.boundary_gradient_stencil = (
            SolenoidalProjection._boundary_gradient_stencil(cells)
        )
        self.boundary_normals = np.zeros(cells.shape + (2,))
        for c in cells.flat:
            match c:
                case cell.BoundaryCell(index=index, normal=normal):
                    self.boundary_normals[index] = normal / np.sum(np.abs(normal))

    def project(self, w: NDArray[np.float_]):
        b = self._projection_b(w)
        # Can extract residuals from this solve method for debugging.
        x = self.multigrid_solver.solve(b)
        P = np.zeros(shape=self.cells.shape)
        P[self.indices.fluid] = x
        for c in self.cells.flat:
            match c:
                case cell.BoundaryCell(index=index, normal=normal, x_diff=xd, y_diff=yd):
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
                    case cell.FluidCell(num=num):
                        row[num] = 1
                    case cell.BoundaryCell(normal=normal, x_diff=x_diff, y_diff=y_diff):
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
        s = cell.central_difference_stencil(self.cells.shape)
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
        s = cell.central_difference_stencil(cells.shape)
        id = np.full(cells.shape, 2)
        jd = np.full(cells.shape, 2)
        for c in cells.flat:
            if not isinstance(c, cell.BoundaryCell):
                continue
            if c.x_diff:
                if isinstance(c.left, cell.FluidCell):
                    s.i_right[c.index] = c.i
                else:
                    s.i_left[c.index] = c.i
                id[c.index] = 1
            if c.y_diff:
                if isinstance(c.down, cell.FluidCell):
                    s.j_up[c.index] = c.j
                else:
                    s.j_down[c.index] = c.j
                jd[c.index] = 1
            # Handle concave corners, where a flat boundary is next to an inside obstacle cell,
            # like so (center is the horizontal, flat, boundary):
            #
            #    1,1,1
            #    1,1,1
            #    0,0,1
            #
            if not c.x_diff:
                if isinstance(c.left, cell.ObstacleInteriorCell):
                    s.i_left[c.index] = c.i
                    id[c.index] = 1
                if isinstance(c.right, cell.ObstacleInteriorCell):
                    s.i_right[c.index] = c.i
                    id[c.index] = 1
            if not c.y_diff:
                if isinstance(c.up, cell.ObstacleInteriorCell):
                    s.j_up[c.index] = c.j
                    jd[c.index] = 1
                if isinstance(c.down, cell.ObstacleInteriorCell):
                    s.j_down[c.index] = c.j
                    jd[c.index] = 1

        return cell.FiniteDifferenceStencil(
            j=s.j,
            i=s.i,
            j_up=s.j_up,
            j_down=s.j_down,
            i_right=s.i_right,
            i_left=s.i_left,
            jd=jd,
            id=id,
        )