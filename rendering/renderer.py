from moderngl_window import geometry
from PIL import Image
import numpy as np
from numpy.typing import NDArray


class Renderer:

    noop_vertex_shader = """
    #version 410
    in vec2 in_position;

    void main() {
        gl_Position = vec4(in_position, 0.0, 1.0);
    }
    """

    def __init__(self, ctx, grid, resolution):
        self.ctx = ctx
        with open("rendering/advection.glsl", "r") as file:
            advection_fragment_shader = file.read()
        with open("rendering/display.glsl", "r") as file:
            display_fragment_shader = file.read()
        self.prog_advection = ctx.program(
            vertex_shader=self.noop_vertex_shader,
            fragment_shader=advection_fragment_shader,
        )
        self.prog_display = self.ctx.program(
            vertex_shader=self.noop_vertex_shader,
            fragment_shader=display_fragment_shader,
        )
        rows, columns = grid.shape
        self.prog_advection["rows"].value = rows
        self.prog_advection["columns"].value = columns
        self.prog_advection["resolution"].value = resolution
        self.prog_display["rows"].value = rows
        self.prog_display["columns"].value = columns
        self.prog_display["resolution"].value = resolution

        # Full-screen quad geometry.
        self.quad = geometry.quad_fs(normals=False, uvs=False)

        # Texture format specs:
        # https://moderngl.readthedocs.io/en/latest/topics/texture_formats.html
        self.velocityFieldTexture = self.ctx.texture(
            (columns, rows),
            2,
            dtype="f4",
            data=np.zeros(grid.shape + (2,), dtype=np.float32).tobytes(),
        )
        self.obstacleTexture = self.ctx.texture(
            (columns, rows),
            1,
            dtype="u1",
            data=grid.astype(np.uint8).tobytes(),
        )
        waterImg = Image.open("rendering/water_texture.jpg").resize(size=resolution)

        self.advect_tex1 = self.ctx.texture(waterImg.size, 3, waterImg.tobytes())
        self.advect_tex2 = self.ctx.texture(waterImg.size, 3, waterImg.tobytes())

        self.current_tex = self.advect_tex1
        self.target_tex = self.advect_tex2

        self.fbo_1 = self.ctx.framebuffer(color_attachments=[self.advect_tex1])
        self.fbo_2 = self.ctx.framebuffer(color_attachments=[self.advect_tex2])

        self.current_fbo = self.fbo_1
        self.target_fbo = self.fbo_2

        self.prog_advection["fluidTexture"].value = 0
        self.prog_display["fluidTexture"].value = 0
        self.prog_advection["vectorFieldTexture"].value = 1
        self.prog_display["obstacleTexture"].value = 2
        self.velocityFieldTexture.use(location=1)
        self.obstacleTexture.use(location=2)

    def render(self, w: NDArray, dt: float):
        # Render advected fluid from the last fluid state, into the target FBO.
        self.target_fbo.use()
        self.current_tex.use(location=0)
        self.prog_advection["dt"].value = dt
        self.quad.render(self.prog_advection)
        # Now switch rendering to the screen, and render the newly advected fluid.
        self.ctx.screen.use()
        self.target_tex.use(location=0)
        self.quad.render(self.prog_display)

        # Swap buffers such that the newly advected fluid is used
        # as the source for the next advection.
        self.current_tex, self.target_tex = self.target_tex, self.current_tex
        self.current_fbo, self.target_fbo = self.target_fbo, self.current_fbo

        self.velocityFieldTexture.write(w.astype(np.float32).tobytes())

