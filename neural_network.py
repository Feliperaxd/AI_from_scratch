import json
import numpy as np
from enum import Enum
from typing import List
from activation_module import ActivationFunctions

class Utils(Enum):

    EPSILON = np.finfo(float).eps
    LEARNING_RATE = 0.01

class Layer:


    def __init__(
        self: 'Layer',
        n_inputs: int,
        n_neurons: int,
    ) -> None:

        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def foward(
        self: 'Layer',
        inputs: np.ndarray
    ) -> np.ndarray:

        self.inputs = inputs
        self.outputs = np.dot(
            a=inputs, 
            b=self.weights
        ) + self.biases

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
        self.loss = -np.sum(
            a=one_hot_vector * np.log(predicted_output)
        ) / len(one_hot_vector)

        return self.loss

    def backward_propagation(
        self: 'Layer',
        one_hot_vector: np.ndarray
    ) -> None:

        delta_output = (self.outputs - one_hot_vector) / (len(one_hot_vector) - 1)

        grad_weights = np.outer(
            a=self.inputs, 
            b=delta_output
        )
        grad_biases = np.sum(
            a=delta_output, 
            axis=0, 
            keepdims=True
        )

        self.weights -= Utils.LEARNING_RATE.value * grad_weights
        self.biases -= Utils.LEARNING_RATE.value * grad_biases


class NeuralNetwork:


    def __init__(
        self: 'NeuralNetwork',
        net_shape: List[tuple(int, int)], #  Layer = (n_inputs, n_neurons)
        activation_funcions: List[ActivationFunctions] #  Each funcion will be used on a layer 
    ) -> None: 

        self.net_shape = net_shape
        self.activation_funcions = activation_funcions
        self.layers = []

        for index, layer_shape in enumerate(self.net_shape):
            with open('weights.json', 'r+') as file:
                data = json.load(file)
                layer_name = f'layer_{index}'

                if not layer_name in data:
                    json.dump(data, layer_name)

            self.layers.append(
                Layer(
                    n_inputs=layer_shape[0],
                    n_neurons=layer_shape[1]
                )
            )

    def save_weights(
        self: 'NeuralNetwork'
    ) -> None:

        with open('weights.json', 'r+') as file:
            for layer in self.layers:
                pass

#:)