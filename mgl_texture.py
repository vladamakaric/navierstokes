import time
import struct
import moderngl_window
import moderngl
from PIL import Image
import numpy as np


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
    in vec2 texcoord;
    out vec4 our_color;
    out vec2 our_texcoord;

    void main() {
        gl_Position = vec4(in_position, 1.0);
        our_color = vec4(in_color, abs(sin(iTime)));
        our_texcoord = texcoord;

    }
    """

    # --- Fragment Shader ---
    fragment_shader = """
    #version 410
    
    in vec4 our_color;
    in vec2 our_texcoord;

    out vec4 color;

    uniform sampler2D t;

    void main() {

        color = texelFetch(t, ivec2(our_texcoord*10), 0);
        color.b=0;
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
                -0.5,
                -0.5,
                0.0,
                0.5,
                -0.5,
                0.0,
                0.0,
                0.5,
                0.0,
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
        self.vbo3 = self.ctx.buffer(
            struct.pack(
                "6f",
                0.0,
                1-0.0,
                1.0,
                1-0.0,
                0.5,
                1-1.0,
            )
        )
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "3f", "in_position"), (self.vbo2, "3f", "in_color"), (self.vbo3, '2f', 'texcoord')]
        )
        img = Image.open('fikus.png')
        self.texture1 = self.ctx.texture(img.size, 3, img.tobytes())
        self.t = np.random.random((10,10,2)).astype(np.float32)

        for j in range(0,10):
            for i in range(0,10):
                if j%3 == 0:
                    self.t[j][i][0] = 1
                    self.t[j][i][1] = 0
                else:
                    self.t[j][i][0] = 0
                    self.t[j][i][1] = 1
        # 10x10 Texture with two 32 bit floats per coordinate.
        self.texture2 = self.ctx.texture((10,10), 2, dtype='f4', data=self.t.tobytes())
        self.texture2.use()


    def on_render(self, t: float, frametime: float):
        current_time = time.time() - self.start_time
        self.prog["iTime"].value = current_time
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render()

        print(frametime)
        pfp = np.floor(current_time*10)
        for j in range(0,10):
            for i in range(0,10):
                if j%3 == (pfp%3):
                    self.t[j][i][0] = 1
                    self.t[j][i][1] = 0
                else:
                    self.t[j][i][0] = 0
                    self.t[j][i][1] = 1
        self.texture2.write(self.t.tobytes())


if __name__ == "__main__":
    moderngl_window.run_window_config(ShadertoyWindow)
