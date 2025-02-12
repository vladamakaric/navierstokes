import time
import moderngl
import moderngl_window
from moderngl_window import geometry
import numpy as np
import navier_stokes


def read_matrix(filename):
    with open(filename) as f:
        return np.array([list(map(int, line.strip().split(","))) for line in f])


class SimulationWindow(moderngl_window.WindowConfig):
    # Request an OpenGL 4.1 core context.
    gl_version = (4, 1)
    title = "Navier Stokes Simulation"
    window_size = (1600, 1600)
    # Disable fixed aspect raio ctx.viewport.
    aspect_ratio = None
    resizable = False

    vertex_shader = """
    #version 410
    in vec2 in_position;

    void main() {
        gl_Position = vec4(in_position, 0.0, 1.0);
    }
    """
    # --- Fragment Shader ---

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mouse_pos = (0, 0)
        with open("vector_field.glsl", "r") as file:
            glorious_line_fragment_shader = file.read()
        self.prog = self.ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=glorious_line_fragment_shader,
        )
        self.grid = read_matrix("grids/rectangle20x20.txt")
        height, width = self.grid.shape
        self.prog["rows"].value = height
        self.prog["columns"].value = width
        self.prog["resolution"].value = (self.window_size[0], self.window_size[1])
        # Create a full-screen quad geometry.
        self.quad = geometry.quad_fs(normals=False, uvs=False)
        self.velocityField = np.zeros(self.grid.shape + (2,), dtype=np.float32)
        self.simulator = navier_stokes.Simulator(self.grid)
        # Texture format specs:
        # https://moderngl.readthedocs.io/en/latest/topics/texture_formats.html
        self.velocityFieldTexture = self.ctx.texture(
            (width, height), 2, dtype="f4", data=self.velocityField.tobytes()
        )
        self.obstacleTexture = self.ctx.texture(
            (width, height),
            1,
            dtype="u1",
            data=self.grid.astype(np.uint8).tobytes(),
        )
        self.prog["obstacleTexture"].value = 1
        self.prog["vectorFieldTexture"].value = 0
        self.velocityFieldTexture.use(location=0)
        self.obstacleTexture.use(location=1)

    # def on_render(self, time_delta):
    def on_render(self, t: float, frametime: float):
        print(frametime)
        velocity_field = np.copy(self.simulator.step(dt=frametime))
        max_norm = np.max(np.linalg.norm(velocity_field, axis=2))
        if max_norm:
            velocity_field /= max_norm
        # for index in np.ndindex((velocity_field.shape[0], velocity_field.shape[1])):
        #     l = np.linalg.norm(velocity_field[index])
        #     if l:
        #         velocity_field[index] /= l

        # self.velocityField = velocity_field.astype(np.float32)
        self.velocityFieldTexture.write(velocity_field.astype(np.float32).tobytes())
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        # Render the full-screen quad.
        self.quad.render(self.prog)

    def on_mouse_position_event(self, x, y, dx, dy):
        # moderngl-window's origin is usually at the top-left, but many shaders expect bottom-left.
        # You might need to flip the y-coordinate depending on your needs.
        self.mouse_pos = (x, self.window_size[1] - y)
        print("Mouse position:", x, y, dx, dy)

    # TODO: other mouse events: https://moderngl-window.readthedocs.io/en/latest/guide/basic_usage.html#mouse-input

    # def on_mouse_drag_event(self, x, y, dx, dy):
    #     print("Mouse drag:", x, y, dx, dy)

    # def on_mouse_scroll_event(self, x_offset: float, y_offset: float):
    #     print("Mouse wheel:", x_offset, y_offset)

    def on_mouse_press_event(self, x, y, button):
        for j in range(3, 12):
            for i in range(5, 9):
                self.simulator.force_field[(j, i)] = [0.1, 0]
        print("Mouse button {} pressed at {}, {}".format(button, x, y))

    def on_mouse_release_event(self, x: int, y: int, button: int):
        self.simulator.force_field = np.zeros(shape=self.simulator.force_field.shape)

        print("Mouse button {} released at {}, {}".format(button, x, y))


if __name__ == "__main__":
    moderngl_window.run_window_config(SimulationWindow)
