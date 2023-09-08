import numpy as np
from typing import Optional, Union
from utils import ActivationFunctions, Normalizers, LearningRateManager

class Layer:


    def __init__(
        self: 'Layer',
        weights: np.ndarray,
        biases: np.ndarray,
        activator: ActivationFunctions,
        learning_rate: Union[LearningRateManager, float],
        normalizer: Optional[Normalizers] = None
    ) -> None:
        
        self.weights = weights
        self.biases = biases
        self.activator = activator
        self.learning_rate = learning_rate
        self.normalizer = normalizer

        self.n_inputs = np.size(weights, axis=0)
        self.n_neurons = np.size(weights, axis=1)

        self.inputs = None
        self.outputs = None
        self.grad_weights = None
        self.grad_biases = None
        self.grad_outputs = None

    def foward(
        self: 'Layer',
        inputs: np.ndarray
    ) -> np.ndarray:

        self.inputs = inputs
        self.outputs = np.dot(
            a=inputs, 
            b=self.weights
        ) + self.biases
        
        return self.outputs.flatten()

    def backward(
        self: 'Layer',
        pred_outputs: np.ndarray,
        one_hot_vector: np.ndarray
    ) -> np.ndarray:

        self.grad_outputs = pred_outputs - one_hot_vector
        
        self.grad_weights = np.outer(
            a=self.inputs, 
            b=self.grad_outputs
        )
        self.grad_biases = np.sum(
            a=self.grad_outputs, 
            axis=0, 
            keepdims=True
        )
        return self.grad_outputs
        
    def update(
        self: 'Layer',
        learning_rate: int, 
    ) -> None:
            
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases

    def activation(
        self: 'Layer',
        inputs: np.ndarray
    ) -> np.ndarray:

        self.outputs = self.activator(inputs)
        return self.outputs

    def normalization(
        self: 'Layer',
        inputs: np.ndarray
    ) -> np.ndarray:
        
        if self.normalizer is None:
            outputs = inputs
        else:
            outputs = self.normalizer(inputs)   

        return outputs
#:)