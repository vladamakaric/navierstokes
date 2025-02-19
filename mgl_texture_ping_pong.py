import time
import moderngl_window
from PIL import Image
from moderngl_window import geometry


class ShadertoyWindow(moderngl_window.WindowConfig):
    gl_version = (4, 1)
    title = "Shadertoy Shader in ModernGL"
    window_size = (1920, 1080)
    resizable = False

    vertex_shader = """
    #version 410
    in vec2 in_position;

    void main() {
        gl_Position = vec4(in_position, 0.0, 1.0);
    }
    """

    # --- Fragment Shader ---
    fragment_shader_advect = """
    #version 410
    
    out vec4 color;

    uniform sampler2D t;

    void main() {
        // Toy "velocity field".
        float vx = (gl_FragCoord.x / 1920) - 0.5;
        color = texture(t, (gl_FragCoord.xy + vec2(vx*vx*10 +2,0))/vec2(1920,1080));
    }
    """

    fragment_shader_display = """
    #version 410
    
    out vec4 color;

    uniform sampler2D t;

    void main() {
        color = texture(t, gl_FragCoord.xy/vec2(1920,1080));
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_time = time.time()
        self.quad = geometry.quad_fs(normals=False, uvs=False)
        self.prog = self.ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=self.fragment_shader_advect,
        )
        self.prog_display = self.ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=self.fragment_shader_display,
        )
        img = Image.open("fikus.png").resize(size=(1920, 1080))
        print(img.size)
        self.texture1 = self.ctx.texture(img.size, 3, img.tobytes())
        self.texture2 = self.ctx.texture(img.size, 3, img.tobytes())

        self.fbo_1 = self.ctx.framebuffer(color_attachments=[self.texture1])
        self.fbo_2 = self.ctx.framebuffer(color_attachments=[self.texture2])

        self.current_tex = self.texture1
        self.current_fbo = self.fbo_1
        self.target_tex = self.texture2
        self.target_fbo = self.fbo_2

    def on_render(self, t: float, frametime: float):
        # Render to this fbo.
        self.target_fbo.use()
        self.current_tex.use()
        self.quad.render(self.prog)
        # Now switch rendering to the screen.
        self.ctx.screen.use()
        self.target_tex.use()
        self.quad.render(self.prog_display)

        self.current_tex, self.target_tex = self.target_tex, self.current_tex
        self.current_fbo, self.target_fbo = self.target_fbo, self.current_fbo


if __name__ == "__main__":
    moderngl_window.run_window_config(ShadertoyWindow)
