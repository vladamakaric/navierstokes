import time
import moderngl
import moderngl_window
from moderngl_window import geometry
import numpy as np
import navier_stokes
from PIL import Image

grid = navier_stokes.read_matrix("grids/largebullet20x60.txt")
cell_size = 800 // grid.shape[0]
height = grid.shape[0] * cell_size
width = grid.shape[1] * cell_size

class SimulationWindow(moderngl_window.WindowConfig):
    # Request an OpenGL 4.1 core context.
    gl_version = (4, 1)
    title = "Navier Stokes Simulation"
    window_size = (width, height)
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

    fragment_shader_display = """
    #version 410

    uniform sampler2D fluidTexture;
    uniform ivec2 resolution;
    
    out vec4 color;

    void main() {
        color = vec4(texture(fluidTexture, gl_FragCoord.xy/vec2(resolution)).xyz,1.0);
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
        with open("fluid_advection.glsl", "r") as file:
            fluid_advection_fragment_shader = file.read()
        self.prog = self.ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=glorious_line_fragment_shader,
        )
        self.prog_advect = self.ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=fluid_advection_fragment_shader,
        )
        self.prog_display = self.ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=self.fragment_shader_display,
        )
        # This one is ready to be shipped.
        self.grid = grid
        # self.grid = navier_stokes.read_matrix("grids/largebullet25x50.txt")
        rows, columns = self.grid.shape
        self.prog["rows"].value = rows
        self.prog["columns"].value = columns
        self.prog["resolution"].value = (self.window_size[0], self.window_size[1])
        self.prog_advect["rows"].value = rows
        self.prog_advect["columns"].value = columns
        self.prog_advect["resolution"].value = (self.window_size[0], self.window_size[1])
        self.prog_display["resolution"].value = (self.window_size[0], self.window_size[1])
        # Create a full-screen quad geometry.
        self.quad = geometry.quad_fs(normals=False, uvs=False)
        self.velocityField = np.zeros(self.grid.shape + (2,), dtype=np.float32)
        self.velocityFieldArrows = np.zeros(self.grid.shape + (2,), dtype=np.float32)
        self.P = np.zeros(self.grid.shape, dtype=np.float32)
        self.simulator = navier_stokes.Simulator(self.grid)
        # Texture format specs:
        # https://moderngl.readthedocs.io/en/latest/topics/texture_formats.html
        self.PTexture = self.ctx.texture(
            (columns, rows), 1, dtype="f4", data=self.P.tobytes()
        )
        self.velocityFieldTexture = self.ctx.texture(
            (columns, rows), 2, dtype="f4", data=self.velocityField.tobytes()
        )
        self.velocityFieldArrowsTexture = self.ctx.texture(
            (columns, rows), 2, dtype="f4", data=self.velocityFieldArrows.tobytes()
        )
        self.obstacleTexture = self.ctx.texture(
            (columns, rows),
            1,
            dtype="u1",
            data=self.grid.astype(np.uint8).tobytes(),
        )
        noiseImg = Image.open('water.jpg').resize(size=self.window_size)
        self.noiseTexture = self.ctx.texture(noiseImg.size, 3, noiseImg.tobytes())

        self.advect_tex1 = self.ctx.texture(noiseImg.size, 3, noiseImg.tobytes())
        self.advect_tex2 = self.ctx.texture(noiseImg.size, 3, noiseImg.tobytes())

        self.current_tex = self.advect_tex1
        self.target_tex = self.advect_tex2

        self.fbo_1 = self.ctx.framebuffer(color_attachments=[self.advect_tex1])
        self.fbo_2 = self.ctx.framebuffer(color_attachments=[self.advect_tex2])

        self.current_fbo = self.fbo_1
        self.target_fbo = self.fbo_2

        self.prog["obstacleTexture"].value = 1
        self.prog["vectorFieldTexture"].value = 0
        self.prog["velocityFieldArrows"].value = 4
        self.prog["PTexture"].value = 5
        self.prog["noiseTexture"].value = 2
        self.prog["mouse_px"].value = self.mouse_pos
        self.prog["mouse_box_size_px"].value = self.mouse_box_size
        self.prog_advect["vectorFieldTexture"].value = 0
        self.prog_advect["fluidTexture"].value = 3
        self.prog_display["fluidTexture"].value = 3
        self.velocityFieldTexture.use(location=0)
        self.PTexture.use(location=5)
        self.velocityFieldArrowsTexture.use(location=4)
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

        # Enable blending
        self.ctx.enable(moderngl.BLEND)

        # Set the blend function to standard alpha blending
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    def on_render(self, t: float, frametime: float):
        force_field = np.zeros(shape=self.simulator.cells.shape + (2,))
        if self.mouse_pressed:
            for cell in self.simulator.cells.flat:
                # if not isinstance(cell, navier_stokes.FluidCell):
                #     continue
                # 80:20 is good.
                if isinstance(cell, navier_stokes.ObstacleInteriorCell):
                    continue
                force_field[cell.index] = [80,0]
        P = self.simulator.step(dt=frametime, force_field=force_field)
        OgP = np.copy(P)
        pmin = np.inf
        pmax = -np.inf
        for cell in self.simulator.cells.flat:
            if isinstance(cell, navier_stokes.ObstacleInteriorCell):
                continue
            pmin = np.min([P[cell.index], pmin])
            pmax = np.max([P[cell.index], pmax])
        P = P - pmin
        if pmax - pmin > 0.00001:
            P = P / (pmax - pmin)
        for cell in self.simulator.cells.flat:
            if isinstance(cell, navier_stokes.ObstacleInteriorCell):
                P[cell.index] = 0
            

        # P = P - np.min(P)
        # maxP = np.max(P)
        # if maxP > 0.001:
        #     # Normalize P.
        #     P = P/maxP
        # self.P = P.astype(np.float32)

        # if np.floor(t) % 2 == 1:
            # norms = np.linalg.norm(velocity_field, axis=2).flatten()
            # k = 5
            # topkInd = np.argpartition(norms, -k)[-k:]
            # # print(f"Projection error: {residuals[-1]} after {len(residuals)} iters.")
            # print(f"Largest velocities: {norms[topkInd]}")
            # print(f"FPS: {1/frametime}")
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
        # self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        #############################
        self.target_fbo.use()
        self.current_tex.use(location=3)

        self.prog_advect["dt"].value = frametime
            # self.prog_advect["dt"].value = 0.05231

        self.quad.render(self.prog_advect)
        # Now switch rendering to the screen.
        self.ctx.screen.use()
        self.target_tex.use(location=3)
        self.quad.render(self.prog_display)

        self.current_tex, self.target_tex = self.target_tex, self.current_tex
        self.current_fbo, self.target_fbo = self.target_fbo, self.current_fbo
#  #
        self.quad.render(self.prog)

        ####################

        self.velocityFieldArrows = self.simulator.velocity_field.astype(np.float32)
        # indices = (18:8, 24:30)
        # down, up = 12,16
        # left, right = 24,30

        # print(f' OGP: {OgP[15, 26]}, {OgP[15, 27]}, {OgP[15,28]}')
        # print(f' u [15,27] = {self.simulator.velocity_field}')

        # # print(P[down:up, left:right])


        # for j,i in np.ndindex((self.velocityFieldArrows.shape[0], self.velocityFieldArrows.shape[1])):
        #     if (down > j or j >= up) or (left > i or i >= right):
        #         self.velocityFieldArrows[j,i] = [0,0]

        # # self.velocityFieldArrows[12:18,24:30] = 
        max_norm = np.max(np.linalg.norm(self.velocityFieldArrows, axis=2))
        if max_norm > 0.001:
            self.velocityFieldArrows /= max_norm

        ##################

        self.velocityFieldTexture.write(self.simulator.velocity_field.astype(np.float32).tobytes())
        self.velocityFieldArrowsTexture.write(self.velocityFieldArrows.tobytes())
        self.PTexture.write(P.astype(np.float32).tobytes())

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

    def on_key_event(self, key, action, modifiers):
    # The keys are accessible via self.wnd.keys
        if key == self.wnd.keys.F:
            if action == self.wnd.keys.ACTION_PRESS:
                self.mouse_pressed = True
                print("F key pressed")
            elif action == self.wnd.keys.ACTION_RELEASE:
                self.mouse_pressed = False
                print("F key released")

    def on_mouse_drag_event(self, x, y, dx, dy):
        self.mouse_pos = [x, self.window_size[1] - y]
        self.prog["mouse_px"].value = self.mouse_pos

    def on_mouse_press_event(self, x, y, button):
        self.mouse_pressed = True

    def on_mouse_release_event(self, x: int, y: int, button: int):
        self.mouse_pressed = False


if __name__ == "__main__":
    
    moderngl_window.run_window_config(SimulationWindow)
