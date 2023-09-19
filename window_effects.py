import time
import tkinter as tk
from typing import Optional, Union, Tuple


class WindowEffects:


    @staticmethod
    def static_fade_in(
        root: tk.Tk,
        delta : Union[float, int],
        after_time: Union[float, int],
        position: Optional[Tuple[int, int]] = None,
        max_alpha: Optional[Union[float, int]] = 1.5, 
        start_alpha: Optional[Union[float, int]] = 0
    ) -> None:

        if position is not None:
            root.geometry(f'+{position[0]}+{position[1]}')
            
        alpha = start_alpha
        while alpha <= max_alpha:
            time.sleep(after_time)
            root.attributes('-alpha', alpha)
            alpha += delta 
            root.update()
            
    @staticmethod
    def static_fade_out(
        root: tk.Tk,
        delta : Union[float, int],
        after_time: Union[float, int],
        position: Optional[Tuple[int, int]] = None,
        min_alpha: Optional[Union[float, int]] = 0,
        start_alpha: Optional[Union[float, int]] = 1.5
    ) -> None:

        if position is not None:
            root.geometry(f'+{position[0]}+{position[1]}')
            
        alpha = start_alpha
        while alpha >= min_alpha:
            time.sleep(after_time)
            root.attributes('-alpha', alpha)
            alpha -= delta 
            root.update()

        root.destroy()

