import time
import moderngl
import moderngl_window
from moderngl_window import geometry
import numpy as np
import navier_stokes
from PIL import Image


class SimulationWindow(moderngl_window.WindowConfig):
    # Request an OpenGL 4.1 core context.
    gl_version = (4, 1)
    title = "Navier Stokes Simulation"
    window_size = (1600, 800)
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
        self.mouse_pos = np.array([0, 0])
        self.mouse_box_size = np.array([600, 1400])
        self.mouse_pressed = False
        with open("vector_field.glsl", "r") as file:
            glorious_line_fragment_shader = file.read()
        self.prog = self.ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=glorious_line_fragment_shader,
        )
        # This one is ready to be shipped.
        self.grid = navier_stokes.read_matrix("grids/largebullet20x40.txt")
        # self.grid = navier_stokes.read_matrix("grids/largebullet25x50.txt")
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
        noiseImg = Image.open("noise2.jpg")
        # noiseImg = Image.open('noise2.jpg')
        self.noiseTexture = self.ctx.texture(noiseImg.size, 3, noiseImg.tobytes())
        self.prog["obstacleTexture"].value = 1
        self.prog["vectorFieldTexture"].value = 0
        self.prog["noiseTexture"].value = 2
        self.prog["mouse_px"].value = self.mouse_pos
        self.prog["mouse_box_size_px"].value = self.mouse_box_size
        self.velocityFieldTexture.use(location=0)
        self.obstacleTexture.use(location=1)
        self.noiseTexture.use(location=2)
        # Drawing simple lines for debugging.
        self.lines_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_position;
                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(1.0, 0.0, 0.0, 1.0);  // Red lines
                }
            """,
        )

    # def on_render(self, time_delta):
    def on_render(self, t: float, frametime: float):
        # print(frametime)
        force_field = np.zeros(shape=self.simulator.cells.shape + (2,))
        if self.mouse_pressed:
            # cellSize = min(
            #     np.floor(self.window_size[0] / self.grid.shape[1]),
            #     np.floor(self.window_size[1] / self.grid.shape[0]),
            # )
            # left = max(self.mouse_pos[0] - self.mouse_box_size[0] / 2, 0)
            # right = min(
            #     self.mouse_pos[0] + self.mouse_box_size[0] / 2, self.window_size[0]
            # )
            # bottom = max(self.mouse_pos[1] - self.mouse_box_size[1] / 2, 0)
            # top = min(
            #     self.mouse_pos[1] + self.mouse_box_size[1] / 2, self.window_size[1]
            # )
            for cell in self.simulator.cells.flat:
                if isinstance(cell, navier_stokes.ObstacleInteriorCell):
                    continue
                force_field[cell.index] = [5,0]

            # for j in range
            # for j in range(
            #     int(np.floor(bottom / cellSize)), int(np.floor(top / cellSize))
            # ):
            #     for i in range(
            #         int(np.floor(left / cellSize)), int(np.floor(right / cellSize))
            #     ):
            #         force_field[j][i] = [20, 0]

        # residuals = []
        # velocity_field = np.copy(
        self.simulator.step(dt=frametime, force_field=force_field)
        # )

        if np.floor(t) % 2 == 1:
            # norms = np.linalg.norm(velocity_field, axis=2).flatten()
            # k = 5
            # topkInd = np.argpartition(norms, -k)[-k:]
            # # print(f"Projection error: {residuals[-1]} after {len(residuals)} iters.")
            # print(f"Largest velocities: {norms[topkInd]}")
            print(f"FPS: {1/frametime}")
        # if residuals[-1] > 1e-5:
        #     norms = np.linalg.norm(velocity_field, axis=2).flatten()
        #     k = 5
        #     topkInd = np.argpartition(norms, -k)[-k:]
        #     print(f"Projection error: {residuals[-1]} after {len(residuals)} iters.")
        #     print(f"Largest velocities: {norms[topkInd]}")
        # max_norm = np.max(np.linalg.norm(velocity_field, axis=2))
        # if max_norm:
        #     velocity_field /= max_norm
        # for index in np.ndindex((velocity_field.shape[0], velocity_field.shape[1])):
        #     l = np.linalg.norm(velocity_field[index])
        #     if l:
        #         velocity_field[index] /= l

        # self.velocityField = velocity_field.astype(np.float32)
        self.velocityFieldTexture.write(self.simulator.velocity_field.astype(np.float32).tobytes())
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.quad.render(self.prog)
        # TODO: Same way you are drawing lines here, draw particles moving through the fluid.
        # TODO: Or even better, try advecting a whole water texture. I think Stam talked about this.
        # self.renderStreamlinesNearBoundary(velocity_field, dt=frametime)

    def renderStreamlinesNearBoundary(self, velocity_field, dt):
        all_lines = []

        def normalize(p):
            width, height = self.grid.shape[1], self.grid.shape[0]
            x = ((p[0] + 0.5) / width) * 2 - 1
            y = ((p[1] + 0.5) / height) * 2 - 1
            return [x, y]

        for cell in self.simulator.cells.flat:
            # if isinstance(cell, navier_stokes.ObstacleInteriorCell):
            #     continue
            if not isinstance(cell, navier_stokes.FluidCell):
                continue
            all_fluid = True
            for n in cell.neighbors:
                if not isinstance(n, navier_stokes.FluidCell):
                    all_fluid = False
                    break
            if all_fluid:
                continue
            _, path = navier_stokes.trace(
                pos=np.array([cell.i, cell.j], dtype=np.float64),
                dt=dt * 50,
                steps=50,
                velocity_field=velocity_field,
                # dir=-1,
                savePath=True,
            )
            path = [np.array([cell.i, cell.j])] + path
            for i in range(0, len(path) - 1):
                p1 = normalize(path[i])
                p2 = normalize(path[i + 1])
                all_lines += p1
                all_lines += p2

        lines_buffer = self.ctx.buffer((np.array(all_lines, dtype="f4").tobytes()))
        lines_vao = self.ctx.simple_vertex_array(
            self.lines_prog, lines_buffer, "in_position"
        )
        lines_vao.render(mode=moderngl.LINES)

    def on_mouse_position_event(self, x, y, dx, dy):
        # moderngl-window's origin is usually at the top-left, but many shaders expect bottom-left.
        # You might need to flip the y-coordinate depending on your needs.
        self.mouse_pos = [x, self.window_size[1] - y]
        self.prog["mouse_px"].value = self.mouse_pos

    # TODO: other mouse events: https://moderngl-window.readthedocs.io/en/latest/guide/basic_usage.html#mouse-input

    def on_mouse_drag_event(self, x, y, dx, dy):
        self.mouse_pos = [x, self.window_size[1] - y]
        self.prog["mouse_px"].value = self.mouse_pos

    def on_mouse_press_event(self, x, y, button):
        self.mouse_pressed = True

    def on_mouse_release_event(self, x: int, y: int, button: int):
        self.mouse_pressed = False


if __name__ == "__main__":
    moderngl_window.run_window_config(SimulationWindow)
