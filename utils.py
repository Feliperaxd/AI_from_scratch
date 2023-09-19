import os
import numpy as np
from typing import Optional, Tuple


class Initializers:


    @staticmethod
    def random_weights(
        layer_shape: Tuple[int, int]
    ) -> np.ndarray:
        
        return 0.10 * np.random.randn(layer_shape[0], layer_shape[1])

    @staticmethod
    def zeros_biases(
        layer_shape: Tuple[int, int]
    ) -> np.ndarray:
        
        return np.zeros((1, layer_shape[1]))
    
class LossCalculations:
    
    
    @staticmethod
    def categorical_crossentropy(
        one_hot_vector: np.ndarray, 
        predicted_output: np.ndarray
    ) -> float:

        predicted_output = np.clip(
            a=predicted_output, 
            a_min=np.finfo(float).eps, 
            a_max=1 - np.finfo(float).eps
        )
        loss = -np.sum(
            a=one_hot_vector * np.log(predicted_output)
        ) / len(one_hot_vector)

        return loss

class Normalizers:

    
    @staticmethod
    def minmax(
        inputs: np.ndarray
    ) -> np.ndarray:

        min_value = np.min(inputs)
        max_value = np.max(inputs)
        outputs = (inputs - min_value) / (max_value - min_value)
        
        return outputs

class ActivationFunctions:
    
    
    @staticmethod
    def sigmoid(
        inputs: np.ndarray,
        derivative: Optional[bool] = False
    ) -> np.ndarray:

        if not derivative:
            outputs = 1 / (1 + np.exp(-inputs))

        elif derivative:
            s = 1 / (1 + np.exp(-inputs))
            outputs = s * (1 - s)

        return outputs

    @staticmethod
    def relu(
        inputs: np.ndarray,
        derivative: Optional[bool] = False
    ) -> np.ndarray:

        if not derivative:
            outputs = np.maximum(0, inputs)

        elif derivative:
            outputs = np.where(inputs >= 0, 1, 0)
        
        return outputs

    @staticmethod
    def leaky_relu(
        inputs: np.ndarray,
        slope: Optional[float] = 0.01,
        derivative: Optional[bool] = False
    ) -> np.ndarray:
        
        if not derivative:
            outputs = np.maximum(slope * inputs, inputs)

        elif derivative:
            outputs = np.where(inputs >= 0, 1, slope)

        return outputs
        
    @staticmethod
    def tanh(
        inputs: np.ndarray,
        derivative: Optional[bool] = False
    ) -> np.ndarray:

        if not derivative:
            outputs = np.tanh(inputs)

        elif derivative:
            outputs = 1 - np.tanh(inputs)**2

        return outputs

    @staticmethod
    def softmax(
        inputs: np.ndarray,
        derivative: Optional[bool] = False
    ) -> np.ndarray:

        if not derivative:
            exp_inputs = np.exp(inputs - np.max(inputs))
            outputs = exp_inputs / np.sum(exp_inputs)

        elif derivative:
            num_categories= len(inputs)
            jacobian_matrix = np.zeros((num_categories, num_categories))

            for i in range(num_categories):
                for j in range(num_categories):
                    if i == j:
                        jacobian_matrix[i, j] = inputs[i] * (1 - inputs[i])
                    else:
                        jacobian_matrix[i, j] = -inputs[i] * inputs[j]
            outputs = jacobian_matrix

        return outputs
#:)