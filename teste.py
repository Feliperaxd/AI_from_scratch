import os
import threading
import numpy as np


array = [
    np.ones((1, 3)),
    np.ones((3, 3)),
    np.ones((3, 3)),
]


for i in array:
    print(i)
    print('-' * 50)

def reg(tensor):
    max_matrix_len = len(max(tensor, key=len))

    for i, matrix in enumerate(tensor):
        if len(matrix) < max_matrix_len:
            for j in range(max_matrix_len - len(matrix)):
                tensor[i] = np.concatenate((matrix, np.zeros((1, len(max(matrix, key=len))))))

    return tensor


for i in reg(tensor=array):
    print('*' * 50)
    print(i)
    