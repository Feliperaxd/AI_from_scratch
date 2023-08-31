import numpy as np
from typing import List, Tuple, Optional


"""class TensorTools:
    

    @staticmethod
    def transform(
        list_of_arrays: List[np.ndarray]
    ) -> np.ndarray:
        
        max_matrix_len = len(max(list_of_arrays, key=len))
        
        for index, matrix in enumerate(list_of_arrays):
            if len(matrix) < max_matrix_len:
                for _ in range(max_matrix_len - len(matrix)):
                    list_of_arrays[index] = np.concatenate(
                        (
                            list_of_arrays[index],
                            np.zeros((1, len(matrix[0])))   
                        )
                    )
        return np.array(list_of_arrays)"""



class Tensor:


    def __init__(
        self: 'Tensor'
    ) -> None:
        
        self.shape = None
        self.data_type = None
        self.array_list = []

    def with_zeros(
        self: 'Tensor',
        shape: Tuple[Tuple[int, int]],
        data_type: Optional[np.number] = np.int8
    ) -> None:
        
        for matrix in shape:
            self.array_list.append(
                np.zeros(matrix, dtype=data_type)
            )

    def with_ones(
        self: 'Tensor',
        shape: Tuple[Tuple[int, int]],
        data_type: Optional[np.number] = np.int8
    ) -> None:
        
        for matrix in shape:
            self.array_list.append(
                np.ones(matrix, dtype=data_type)
            )

array = [
    np.ones((1, 3)),
    np.ones((3, 3)),
    np.ones((3, 3)),
]

