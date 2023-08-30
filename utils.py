import os
import numpy as np
from typing import Optional

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
        inputs: np.ndarray
    ) -> np.ndarray:

        return np.exp(-inputs) / ((np.exp(-inputs) + 1) ** 2)

    @staticmethod
    def relu(
        inputs: np.ndarray
    ) -> np.ndarray:

        outputs = np.maximum(0, inputs)
        return outputs

    @staticmethod
    def leaky_relu(
        inputs: np.ndarray,
        slope: Optional[float] = 0.01
    ) -> np.ndarray:
        
        outputs = np.maximum(slope * inputs, inputs)
        return outputs
        
    @staticmethod
    def tanh(
        inputs: np.ndarray
    ) -> np.ndarray:

        outputs = np.tanh(inputs)
        return outputs

    @staticmethod
    def softmax(
        inputs: np.ndarray
    ) -> np.ndarray:

        outputs = np.exp(inputs) / np.sum(np.exp(inputs))           
        return outputs
#:)