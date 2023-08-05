import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Any, Callable, Union
from neural_network import NeuralNetwork
from fake_fruits import FruitsData, FakeFruit
from activation_module import ActivationFunctions


class Model:
    
    
    def __init__(
        self: 'Model'
    ) -> None:
        
        #  Others!
        self.data = None
        self.network = None
        
        #  Metrics!       
        self.score = 0
        self.acuraccy = 0
        self.total_epochs = 0
        self.better_score = [0, 0] #  [Epoch, Score]
        self.better_acuraccy = [0, 0] #  [Epoch, Acuraccy]
        self.training_count = 0

        #  Coordinates!        
        self.score_coord = []
        self.epochs_coord = []
        self.accuracy_coord = []
        
        #  Private!
        self._epoch_count = 0
        self._hit_count = 0
        
    def create(
        self: 'Model',
        shape: List[Tuple[int, int]], #  Layer = (n_inputs, n_neurons)
        activators: List[Union[ActivationFunctions, str]], #  Each funcion will be used on a layer
        data_path: Optional[str] = 'model_data.json'
    ) -> None:
        
        self.data = {}
        self.shape = shape
        self.activators = activators
        
        self.network = NeuralNetwork(
            shape=self.shape,
            activators=self.activators
        )
        self.network.inject_tensors()
        
        with open(data_path, 'w') as file:
            file.seek(0)        
            json.dump(self.data, file)
        
    def save_data(
        self: 'Model',
        data_path: Optional[str] = 'model_data.json'
    ) -> None:
    
        weights = []
        biases = []
        for layer_weights, layer_biases in zip(*self.network.get_tensors()):
            weights.append(layer_weights.tolist())
            biases.append(layer_biases.tolist())
        
        self.data = {
            #  Architecture!
            'shape': self.network.shape,
            'activators': self.network.activators,
            
            #  Metrics!
            'score': self.score,
            'acuraccy': self.acuraccy,
            'total_epochs': self.total_epochs, 
            'better_score': self.better_score,
            'better_acuraccy': self.better_acuraccy,
            'training_count': self.training_count,
            
            #  Coordinates!
            'score_coord': self.score_coord,
            'epochs_coord': self.epochs_coord,
            'accuracy_coord': self.accuracy_coord,

            #  Tensors!
            'weights': weights,
            'biases': biases
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
                        
        #  Metrics!
        self.score = self.data['score']
        self.acuraccy = self.data['acuraccy']
        self.total_epochs = self.data['total_epochs']
        self.better_score = self.data['better_score']
        self.better_acuraccy = self.data['better_acuraccy']
        self.training_count = self.data['training_count']
        
        #  Coordinates!
        self.score_coord = self.data['score_coord']
        self.epochs_coord = self.data['epochs_coord']
        self.accuracy_coord = self.data['accuracy_coord']
        
        self.network = NeuralNetwork(
            shape=self.data['shape'],
            activators=self.data['activators']
        )
        self.network.inject_tensors(
            weights=[np.array(x) for x in self.data['weights']],
            biases=[np.array(x) for x in self.data['biases']]
        )
        
    def training(
        self: 'Model',
        inputs: np.ndarray,
        target: Any,
        output_rule: Callable[[Any], Any],
        one_hot_vector: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        
        outputs = self.foward_propagation(inputs)
        delta_outputs = self.backward_propagation(one_hot_vector)
        output = output_rule(outputs)
        
        self.training_count += 1
        self._epoch_count += 1    
        
        if target == output:
            self.score += 1
            self._hit_count += 1
        else: 
            self.score -= 1
          
        if self._epoch_count % 1000 == 0:
            self.acuraccy = f'{(self._hit_count / 1000) * 100:.3f}'        
            TERMINAR ISSO NÃO PODE SAIR STRING 
        
        return outputs, delta_outputs, output

    def operate(
        self: 'Model',
        inputs: np.ndarray,
        one_hot_vector: np.ndarray
    ) -> np.ndarray:
        
        outputs = self.foward_propagation(inputs)
        self.backward_propagation(one_hot_vector)
        
        return outputs
    