import json
import numpy as np
from enum import Enum
from typing import List, Optional, Tuple, Union
from activation_module import ActivationFunctions

class Utils(Enum):

    EPSILON = np.finfo(float).eps
    LEARNING_RATE = 0.01

class Layer:


    def __init__(
        self: 'Layer',
        activator: Union[ActivationFunctions, str],
        weights: Union[np.ndarray, List],
        biases: Union[np.ndarray, List]
    ) -> None:
            
        self.activator = activator
        self.weights = np.array(weights)
        self.biases = np.array(biases)

        self.n_inputs = np.size(weights)
        self.n_neurons = np.size(weights[0])

        self.inputs = None
        self.outputs = None
        self.delta_outputs = None

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
        inputs: np.ndarray,
        one_hot_vector: np.ndarray
    ) -> None:

        self.delta_output = (inputs - one_hot_vector) / (len(one_hot_vector) - 1)

        grad_weights = np.outer(
            a=self.inputs, 
            b=self.delta_output
        )
        grad_biases = np.sum(
            a=self.delta_output, 
            axis=0, 
            keepdims=True
        )

        self.weights -= Utils.LEARNING_RATE.value * grad_weights
        self.biases -= Utils.LEARNING_RATE.value * grad_biases

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

    def normalizer(
        self: 'Layer',
        inputs: np.ndarray
    ) -> np.ndarray:

        min_value = np.min(inputs)
        max_value = np.max(inputs)
        self.outputs = (inputs - min_value) / (max_value - min_value)
        
        return self.outputs

    def categorical_crossentropy(
        self: 'Layer', 
        one_hot_vector: np.ndarray, 
        predicted_output: np.ndarray
    ) -> float:

        predicted_output = np.clip(
            a=predicted_output, 
            a_min=Utils.EPSILON.value, 
            a_max=1 - Utils.EPSILON.value
        )
        loss = -np.sum(
            a=one_hot_vector * np.log(predicted_output)
        ) / len(one_hot_vector)

        return loss
    

class NeuralNetwork:


    def __init__(
        self: 'NeuralNetwork',
        shape: List[Tuple[int, int]],
        activators: List[Union[ActivationFunctions, str]],
    ) -> None: 

        self.shape = shape
        self.activators = activators
        
        self.outputs = None
        self.delta_outputs = None
        
        self.layers = []
        self._weight_tensor = []
        self._bias_tensor = []
    
    def inject_tensors(
        self: 'NeuralNetwork',
        weights: Optional[List[np.ndarray]] = None,
        biases: Optional[List[np.ndarray]] = None
    ) -> None:
        
        self.layers.clear()       
        self._weight_tensor.clear()
        self._bias_tensor.clear()
        
        if not self._weight_tensor:
            for layer_shape in self.shape:
                self._weight_tensor.append(
                    0.10 * np.random.randn(layer_shape[0], layer_shape[1])
                )
        else:
            self._weight_tensor = weights
                
        if not self._bias_tensor:
            for layer_shape in self.shape:
                self._bias_tensor.append(
                   np.zeros((1, layer_shape[1]))
                )
        else:
            self._bias_tensor = biases
        
        for layer_shape, layer_activator, layer_weights, layer_biases in zip(
            self.shape, self.activators, self._weight_tensor, self._bias_tensor):
            self.layers.append(
                Layer(
                    activator=layer_activator,
                    weights=layer_weights,
                    biases=layer_biases
                )
            )
    
    def get_tensors(
        self: 'NeuralNetwork'
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        
        self._weight_tensor.clear()
        self._bias_tensor.clear()
        
        for layer in self.layers:
            self._weight_tensor.append(layer.weights)
            self._bias_tensor.append(layer.biases)
        
        return self._weight_tensor, self._bias_tensor
        
    def foward_propagation(
        self: 'NeuralNetwork',
        inputs: np.ndarray
    ) -> np.ndarray:
        
        self.outputs = inputs
        for layer in self.layers:
            self.outputs = layer.normalizer(self.outputs)
            self.outputs = layer.foward(self.outputs)
            self.outputs = layer.activation(self.outputs)

        return self.outputs               

    def backward_propagation(
        self: 'NeuralNetwork',
        one_hot_vector: np.ndarray
    )  -> np.ndarray:
        
        reversed_layers = list(reversed(self.layers))

        for layer in reversed_layers:
            if not self.delta_outputs:
                self.delta_outputs = self.outputs

            self.delta_outputs = layer.backward(
                inputs=self.delta_outputs,
                one_hot_vector=one_hot_vector
            )
#:)