import os
import threading
import numpy as np
from model import Model
import matplotlib.pyplot as plt
from fake_fruits import FakeFruit, FruitsData


class teste:

    def __init__(self) -> None:
        
        self.a = [0, 0, 0, 0, 0, 0]


    def k(self):
        self.a[0] += 1
        self.a[1] += 1
        self.a[2] += 1
        self.a[3] += 1
        self.a[4] += 1
        self.a[5] += 1

t = teste()
threads = []

for i in range(100):
    for i in range(100):
        thread = threading.Thread(
            target=t.k
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

print(t.a)