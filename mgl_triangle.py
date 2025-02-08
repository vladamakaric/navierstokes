import time
import struct
import moderngl_window
import moderngl


class ShadertoyWindow(moderngl_window.WindowConfig):
    # Request an OpenGL 4.1 core context.
    gl_version = (4, 1)
    title = "Shadertoy Shader in ModernGL"
    window_size = (1920, 1080)
    resizable = False

    # --- Vertex Shader ---
    vertex_shader = """
    #version 410
    uniform float iTime;
    in vec3 in_position;
    in vec3 in_color;
    out vec4 our_color;

    void main() {
        gl_Position = vec4(in_position, 1.0);
        our_color = vec4(in_color, abs(sin(iTime)));
    }
    """

    # --- Fragment Shader ---
    fragment_shader = """
    #version 410
    
    in vec4 our_color;
    out vec4 color;

    void main() {
        color = our_color;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_time = time.time()
        self.ctx.enable(moderngl.BLEND)

        # Compile the shader program.
        self.prog = self.ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=self.fragment_shader,
        )
        self.vbo = self.ctx.buffer(
            struct.pack(
                "9f",
                -0.5,-0.5,0.0,
                0.5,-0.5,0.0,
                0.0,0.5,0.0,
            )
        )
        self.vbo2 = self.ctx.buffer(
            struct.pack(
                "9f",
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            )
        )
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "3f", "in_position"), (self.vbo2, "3f", "in_color")]
        )

    def on_render(self, t: float, frametime: float):
        current_time = time.time() - self.start_time
        self.prog["iTime"].value = current_time
        print(self.prog["iTime"].value)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render()


if __name__ == "__main__":
    moderngl_window.run_window_config(ShadertoyWindow)
