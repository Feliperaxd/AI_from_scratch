import json
import numpy as np
from enum import Enum
from typing import List, Optional, Tuple
from activation_module import ActivationFunctions

class Utils(Enum):

    EPSILON = np.finfo(float).eps
    LEARNING_RATE = 0.01

class Layer:


    def __init__(
        self: 'Layer',
        activator: ActivationFunctions,
        weights: np.ndarray,
        biases: np.ndarray
    ) -> None:
            
        self.activator = activator
        self.weights = weights
        self.biases = biases

        self.n_inputs = len(weights)
        self.n_neurons = len(weights[0])

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

        self.outputs = self.activator(inputs)
        
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
        self: 'NeuralNetwork'
    ) -> None: 

        self.data = {}
        self.layers = []
        self.net_shape = []

        self.loss = None
        self.outputs = None
        self.delta_outputs = None

    def create(
        self: 'NeuralNetwork',
        net_shape: List[Tuple[int, int]], #  Layer = (n_inputs, n_neurons)
        activators: List[ActivationFunctions] #  Each funcion will be used on a layer 
    ) -> None:
        
        self.net_shape = net_shape
        with open('model_data.json', 'w') as file:
            json.dump(self.data, file)

        for layer_shape, activator in zip(self.net_shape, activators):
            self.layers.append(
                Layer(
                    activator=activator,
                    weights=0.10 * np.random.randn(layer_shape[0], layer_shape[1]),
                    biases=np.zeros((1, layer_shape[1]))
                )
            )
                
    def save_data(
        self: 'NeuralNetwork',
        data_path: Optional[str] = 'model_data.json'
    ) -> None:
        
        for index, layer in enumerate(self.layers):
            layer_name = f'layer_{index}'
            self.data[layer_name] = { 
                'activator': layer.activator.__qualname__,
                'weights': layer.weights.tolist(),
                'biases': layer.biases.tolist(),
                'n_inputs': layer.n_inputs,
                'n_neurons': layer.n_neurons
            }

        with open(data_path, 'w') as file:
            file.seek(0)        
            json.dump(self.data, file)
        
    def load_data(
        self: 'NeuralNetwork',
        data_path: Optional[str] = 'model_data.json'
    ) -> None:
        
        with open(data_path, 'r') as file:
            self.data = json.load(file)

        self.layers.clear()
        self.net_shape.clear()

        for index in range(len(self.data)):
            layer_name = f'layer_{index}'
            layer = self.data[layer_name]

            self.layers.append(
                Layer(
                    activator=layer['activator'],
                    weights=layer['weights'],
                    biases=layer['biases']
                )
            )
            self.net_shape.append(
                (layer['n_inputs'], layer['n_neurons'])
            )
    
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