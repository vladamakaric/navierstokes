from ipycanvas import Canvas
from IPython.display import display


class BitmapEditor:
    def __init__(self, matrix):
        self.height = matrix.shape[0]
        self.width = matrix.shape[1]
        self.matrix = matrix
        self.cell_size_px = 10
        canvas_width = self.width * self.cell_size_px
        canvas_height = self.height * self.cell_size_px
        self.canvas = Canvas(width=canvas_width, height=canvas_height)
        self.canvas.on_mouse_down(self.mouse_down)
        self.canvas.on_mouse_up(self.mouse_up)
        self.canvas.on_mouse_move(self.handle_mouse_move)
        self.down = False
        self.draw()

    def handle_mouse_move(self, x, y):
        if not self.down:
            return
        row = int(y // self.cell_size_px)
        col = int(x // self.cell_size_px)
        if 0 <= row < self.height and 0 <= col < self.width:
            self.matrix[row][col] = 1
            self.canvas.fill_style = "black"
            self.canvas.stroke_style = "gray"
            self.canvas.fill_rect(
                col * self.cell_size_px,
                row * self.cell_size_px,
                self.cell_size_px,
                self.cell_size_px,
            )
            self.canvas.stroke_rect(
                col * self.cell_size_px,
                row * self.cell_size_px,
                self.cell_size_px,
                self.cell_size_px,
            )

    def mouse_down(self, x, y):
        self.down = True

    def mouse_up(self, x, y):
        self.down = False

    def draw(self):
        for row in range(self.height):
            for col in range(self.width):
                x = col * self.cell_size_px
                y = row * self.cell_size_px
                self.canvas.fill_style = "white"
                self.canvas.fill_rect(x, y, self.cell_size_px, self.cell_size_px)
                self.canvas.stroke_style = "gray"
                self.canvas.stroke_rect(x, y, self.cell_size_px, self.cell_size_px)
        display(self.canvas)
