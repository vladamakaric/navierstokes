import moderngl_window
import numpy as np
import navier_stokes
import rendering


# The 1s in this 0/1 grid represent obstacles for the fluid to flow around.
with open("grids/largebullet.csv") as f:
    grid = np.flip(np.array([list(map(int, line.strip().split(","))) for line in f]), 0)

# For easy math make screen size an integer multiple of grid size,
# such that cells are squares with integer size.
max_window_height = 800
cell_size = max_window_height // grid.shape[0]
height = grid.shape[0] * cell_size
width = grid.shape[1] * cell_size


class SimulationWindow(moderngl_window.WindowConfig):
    gl_version = (4, 1)
    title = "Navier Stokes Simulation"
    window_size = (width, height)
    # Disable fixed aspect ratio.
    aspect_ratio = None
    resizable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_force = False
        self.renderer = rendering.Renderer(self.ctx, grid, resolution=self.window_size)
        # TODO: pass in parameters here, like viscosity.
        self.simulator = navier_stokes.Simulator(grid)

    def on_render(self, t: float, frametime: float):
        self.simulator.step(dt=frametime, force=30 if self.apply_force else 0)
        self.renderer.render(self.simulator.velocity_field, dt=frametime)

    def on_key_event(self, key, action, modifiers):
        if key == self.wnd.keys.F:
            if action == self.wnd.keys.ACTION_PRESS:
                self.apply_force = True
            elif action == self.wnd.keys.ACTION_RELEASE:
                self.apply_force = False


if __name__ == "__main__":

    moderngl_window.run_window_config(SimulationWindow)
