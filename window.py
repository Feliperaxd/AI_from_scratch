from window_effects import *
import tkinter as tk
import time

class PaintWindow():
    

    def __init__(
        self: 'PaintWindow',
        window_size: Tuple[int, int]
    ) -> None:
        
        self.width = window_size[0]
        self.height = window_size[1]
        self.window_size = window_size
        
        self.root = tk.Tk()
        self.root.geometry(f'{self.width}x{self.height}')
        self.root.resizable(False, False)
        
        self.drawing = False
        self.last_x = 0
        self.last_y = 0

        self.canvas = tk.Canvas(
            self.root, bg="white", 
            width=self.width, height=self.height*0.7
        )
        self.canvas.place(x=0, y=0)

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        #  Buttons & Labels!
        _img = tk.PhotoImage(width=1, height=1)
        
        clear_btn = tk.Button(
            self.root, 
            text='cls', 
            command=self.clear_frame
        )
        clear_btn.place(x=0, y=0)

        predict_lbl = tk.Label(
            self.root, text=1,
            width=100, height=70,
            font=("Arial Black", 50, "bold"), 
            image=_img, compound='c'
        )
        predict_lbl.place(
            x=self.width / 2 - 50, 
            y=self.height * 0.70 + (self.height * 0.70 - self.height / 2) - 45
        )


    def start_drawing(
        self: 'PaintWindow', 
        event: tk.Event
    ) -> None:
        
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw(
        self: 'PaintWindow', 
        event: tk.Event
    ) -> None:
        
        x, y = event.x, event.y
        self.canvas.create_line(
            self.last_x, 
            self.last_y, 
            x, y, 
            fill="black", width=5
        )
        self.last_x = x
        self.last_y = y

    def stop_drawing(
        self: 'PaintWindow',
        event: tk.Event
    ) -> None:
        self.drawing = False

    def clear_frame(
        self: 'PaintWindow'
    ) -> None:
        self.canvas.delete('all')
