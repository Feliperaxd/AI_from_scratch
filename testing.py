from neural_network import NeuralNetwork
from window_effects import WindowEffects
from screeninfo import get_monitors
from window import PaintWindow
from PIL import Image, ImageTk, ImageGrab
import tkinter as tk
import numpy as np
import io

screen_size = get_monitors()[0]

model = NeuralNetwork()
model.load_data()
    

window = PaintWindow((500, 500))
WindowEffects.static_fade_in(
    root=window.root, delta=0.05, after_time=0.01, 
    position=(int(screen_size.width / 2 - 250), int(screen_size.height / 2 - 250))
)
window.root.protocol(
    'WM_DELETE_WINDOW', 
    lambda: WindowEffects.static_fade_out(
        root=window.root, delta=0.05, after_time=0.01
    )
)

def save_canvas_as_png():
    # Captura a área do canvas como uma imagem
    x = window.root.winfo_rootx() + window.canvas.winfo_x()
    y = window.root.winfo_rooty() + window.canvas.winfo_y()
    x1 = x + window.canvas.winfo_width()
    y1 = y + window.canvas.winfo_height()
    image = ImageGrab.grab((x, y, x1, y1))

    print(image)

button = tk.Button(window.root, text="Salvar Canvas como PNG", command=save_canvas_as_png)
button.pack()

window.root.mainloop()