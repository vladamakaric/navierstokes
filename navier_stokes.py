import numpy as np
import solenoidal_projection
import cell
from numpy.typing import NDArray


class Simulator:
    """Real time simulator implementing Jos Stam's 1999 'Stable Fluids' paper.

    The state of the fluid is represented by a vector field of the fluid
    velocity. The field is discretized as an equally spaced rectangular grid of
    cells, where some groups of cells may be obstacles for the fluid to flow
    around. However, only simple obstacles are supported, with 45 or 90 degree
    edges, like bullets and valves (see grids directory).

    Each step of the simulator takes the current velocity field and advances it
    in time by dt (time elapsed since the previous step was computed and
    rendered). Four things happen in each iteration, in order:

    1. Apply force. This is just v(t+dt) = v(t) + F*dt. A horizontal force is
       applied to the whole fluid by pressing F.

    2. Advection. This is just velocity transporting velocity. For example, if a
       fluid moving horizontally to the right gets a vertical force applied to
       part of it, that part will get a vertical velocity component, and that
       vertical component will be transported (ie. advected) by the existing
       horizontal component, the same way ink spilled in that same part of the
       fluid would be transported. 

       The method I used is called Semi-Langrangian Advection, this method is
       unconditionally stable, hence Stam's paper name.
    
    3. Diffusion. This is the viscosity term, viscosity is diffusion or
       averaging of velocity. If one layer of fluid moves faster, it will get
       slowed down by neighboring layers and neighboring layers will get sped up
       by it.

    4. Solenoidal Projection. The previous three steps could be viewed as
       integrating the three terms in the Navier Stokes momentum equation:
       1. body force; 2. advection; 3. viscosity. This final step is ensuring
       that the resulting velocity field satisfies the Navier Stokes continuity
       equaiton, ie. that the velocity field has zero divergence. This step also
       ensures that the fluid flows around obstacles. See solenoidal_projection.py
       for details.

       This was the trickiest step to get right, particularly the boundary
       conditions for the fluid to flow around obstacles. I only made it work
       for simple obstacles with 45 and 90 degree angles, like bullets
       and valves. And even there there are some slight issues around the corners,
       some discontinuities which I didn't get to the bottom off. Need to move on to
       other more practical things...
    """
    def __init__(self, grid: NDArray[np.int_], viscosity: float):
        self.cells = cell.create_cell_matrix(grid)
        self.viscosity = viscosity
        self.solenoidal_projection = solenoidal_projection.SolenoidalProjection(
            self.cells
        )
        self.velocity = np.zeros(grid.shape + (2,))
        self.indices = cell.create_indices(self.cells)

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
        self.solenoidal_projection.project(self.velocity)


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
