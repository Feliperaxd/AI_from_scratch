import numpy as np
from utils import ActivationFunctions

class Layer:


    def __init__(
        self: 'Layer',
        weights: np.ndarray,
        biases: np.ndarray,
        activator: ActivationFunctions,
    ) -> None:
        
        self.weights = weights
        self.biases = biases
        self.activator = activator

        self.n_inputs = np.size(weights, axis=0)
        self.n_neurons = np.size(weights, axis=1)

        self.outputs = None
        self._grad_weights = None
        self._grad_biases = None
        self._grad_out = None

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
        one_hot: np.ndarray,
        pred_outputs: np.ndarray
    ) -> np.ndarray:

        self.grad_out = pred_outputs - one_hot
        
        self.grad_weights = np.outer(
            a=self.inputs, 
            b=self.grad_out
        )
        self.grad_biases = np.sum(
            a=self.grad_out, 
            axis=0, 
            keepdims=True
        )
        return self.grad_out
        
    def update(
        self: 'Layer',
        learning_rate: float
    ) -> None:            
        
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases

    def activation(
        self: 'Layer',
        inputs: np.ndarray
    ) -> np.ndarray:

        self.outputs = self.activator(inputs)
        return self.outputs
#:)