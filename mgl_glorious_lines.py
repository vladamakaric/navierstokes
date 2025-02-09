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


# void main(){
#     gl_FragColor = mainImage(gl_FragCoord.xy);
# }

# vertex_shader_quad = """
# attribute vec2 position;
# attribute vec2 texcoord;
# varying vec2 v_texcoord;
# void main()
# {
#     v_texcoord = texcoord;
#     gl_Position = vec4(position, 0.0, 1.0);
# }
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
        glorious_line_fragment_shader = ""
        with open('glorious_lines.glsl', 'r') as file:
            glorious_line_fragment_shader = file.read()
        self.prog = self.ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=glorious_line_fragment_shader,
        )
        # Create a full-screen quad geometry.
        self.quad = geometry.quad_fs()
        self.mouse_pos = (0,0)

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

    # def on_mouse_press_event(self, x, y, button):
    #     print("Mouse button {} pressed at {}, {}".format(button, x, y))

    # def on_mouse_release_event(self, x: int, y: int, button: int):
    #     print("Mouse button {} released at {}, {}".format(button, x, y))


if __name__ == '__main__':
    moderngl_window.run_window_config(ShadertoyWindow)
