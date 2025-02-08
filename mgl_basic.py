import time
import moderngl
import moderngl_window
from moderngl_window import geometry


class ShadertoyWindow(moderngl_window.WindowConfig):
    # Request an OpenGL 4.1 core context.
    gl_version = (4, 1)
    title = "Shadertoy Shader in ModernGL"
    window_size = (1920, 1080)
    resizable = False

    # --- Vertex Shader ---
    vertex_shader = """
    #version 410
    in vec2 in_position;
    void main() {
        // Map vertex positions from [-1,1] to UV coordinates [0,1]
        gl_Position = vec4(in_position, 0.0, 1.0);
    }
    """

    # --- Fragment Shader ---
    fragment_shader = """
    #version 410
    uniform float iTime;
    uniform vec2 iResolution;
    
    out vec4 fragColor;

    // Shadertoy-style mainImage function.
    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        // Normalize the fragment coordinates based on resolution.
        vec2 uv = fragCoord / iResolution;
        // Compute a simple animated color pattern.
        vec3 col = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0.0, 2.0, 4.0));
        fragColor = vec4(col, 1.0);
    }

    void main() {
        // Use gl_FragCoord.xy as the fragment coordinate.
        mainImage(fragColor, gl_FragCoord.xy);
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_time = time.time()
        # Compile the shader program.
        self.prog = self.ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=self.fragment_shader,
        )
        # Create a full-screen quad geometry.
        self.quad = geometry.quad_fs()
        print(self.quad.buffer)

    # def on_render(self, time_delta):
    def on_render(self, t: float, frametime: float):
        # print(frametime)
        # Calculate elapsed time for animation.
        current_time = time.time() - self.start_time
        self.prog["iTime"].value = current_time
        self.prog["iResolution"].value = (self.window_size[0], self.window_size[1])
        # Clear the context.
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        # Render the full-screen quad.
        self.quad.render(self.prog)


if __name__ == '__main__':
    moderngl_window.run_window_config(ShadertoyWindow)
