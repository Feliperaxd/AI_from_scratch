import numpy as np
from typing import Optional, Tuple, Union
from utils import ActivationFunctions, Normalizers

class Layer:


    def __init__(
        self: 'Layer',
        activator: Union[ActivationFunctions, str],
        weights: np.ndarray,
        biases: np.ndarray,
        normalizer: Optional[Union[ActivationFunctions, str]] = None
    ) -> None:
        
        self.activator = activator
        self.weights = weights
        self.biases = biases
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

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
        return self.grad_outputs, self.grad_weights, self.grad_biases
        
    def update(
        self: 'Layer',
        grad_weights: Optional[float] = None,
        grad_biases: Optional[float] = None
    ) -> None:
        
        if grad_weights is not None:
            self.grad_weights = grad_weights
        if grad_biases is not None:    
            self.grad_biases = grad_biases
            
        self.weights -= 1 * self.grad_weights
        self.biases -= 1 * self.grad_biases
        
        self.grad_weights = None
        self.grad_biases = None

    def activation(
        self: 'Layer',
        inputs: np.ndarray
    ) -> np.ndarray:

        if callable(self.activator):
            self.outputs = self.activator(inputs)
        else:
            activation_func = getattr(ActivationFunctions, self.activator)
            self.outputs = activation_func(inputs)

        return self.outputs

    def normalization(
        self: 'Layer',
        inputs: np.ndarray
    ) -> np.ndarray:
        
        if self.normalizer is None:
            return inputs
        
        if callable(self.normalizer):
            outputs = self.normalizer(inputs)
        else:
            normalization_func = getattr(Normalizers, self.normalizer)
            outputs = normalization_func(inputs)
            
        return outputs
#:)