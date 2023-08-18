import numpy as np
from layer import Layer
from typing import List, Optional, Tuple, Union
from utils import ActivationFunctions, Normalizers


class NeuralNetwork:


    def __init__(
        self: 'NeuralNetwork',
        shape: List[Tuple[int, int]],
        activators: List[Union[ActivationFunctions, str]],
        normalizers: Optional[List[Union[Normalizers, str]]] = None
    ) -> None: 

        self.inputs = None
        self.outputs = None
        
        #  Architecture!
        self.shape = shape
        self.activators = activators
        self.normalizers = normalizers
        self.layers = []
        
        #  Private!
        self._weights = []
        self._biases = []
        self._grad_outputs = None
    
    def inject_layers(
        self: 'NeuralNetwork',
        weights: Optional[List[np.ndarray]] = None,
        biases: Optional[List[np.ndarray]] = None
    ) -> None:      
        
        if weights is None:
            self._weights.clear()
            for layer_shape in self.shape:
                self._weights.append(
                    0.10 * np.random.randn(layer_shape[0], layer_shape[1])
                )
        else:
            self._weights = weights
                
        if biases is None:
            self._biases.clear()
            for layer_shape in self.shape:
                self._biases.append(
                   np.zeros((1, layer_shape[1]))
                )
        else:
            self._biases = biases 

        self.layers.clear() 
        for layer_shape, layer_activator, layer_weights, layer_biases in zip(
            self.shape, self.activators, self._weights, self._biases
        ):
            self.layers.append(
                Layer(
                    activator=layer_activator,
                    weights=layer_weights, 
                    biases=layer_biases
                )
            )
        for layer, normalizer in zip(self.layers, self.normalizers):
            layer.normalizer = normalizer
        
    def get_layers_parameters(
        self: 'NeuralNetwork'
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        
        self._weights.clear()
        self._biases.clear()
        
        for layer in self.layers:
            self._weights.append(layer.weights)
            self._biases.append(layer.biases)
        
        return self._weights, self._biases
    
    def update_layers(
        self: 'NeuralNetwork'
    ) -> None:
        
        for layer in self.layers:
            layer.update()
        
    def foward_propagation(
        self: 'NeuralNetwork',
        inputs: np.ndarray
    ) -> np.ndarray:
        
        self.inputs = inputs
        self.outputs = inputs
        for layer in self.layers:
            self.outputs = layer.normalization(self.outputs)
            self.outputs = layer.foward(self.outputs)
            self.outputs = layer.activation(self.outputs)

        return self.outputs               

    def backward_propagation(
        self: 'NeuralNetwork',
        one_hot_vector: np.ndarray
    ) -> None:
        
        reversed_layers = list(reversed(self.layers))
        for layer in reversed_layers:
            if self._grad_outputs is None:
                self._grad_outputs = self.outputs

            self._grad_outputs = layer.backward(
                pred_outputs=self.outputs,
                one_hot_vector=one_hot_vector
            )[2]
#:)