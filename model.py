import json
import numpy as np
from neural_network import NeuralNetwork
from utils import ActivationFunctions, Normalizers
from typing import Any, Callable, List, Optional, Tuple, Union

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
        self.epoch_count = 0
        self.input_count = 0
        self.best_score = (0, 0) #  (Epoch, Score)
        self.best_acuraccy = (0, 0) #  (Epoch, Acuraccy)
        
        #  Histories!        
        self.score_history = []
        self.accuracy_history = []
        
        #  Private!
        self._acuraccy_count = 0
        self._hit_count = 0
        
    def create(
        self: 'Model',
        shape: List[Tuple[int, int]], #  Layer = (n_inputs, n_neurons)
        activators: List[Union[ActivationFunctions, str]], #  Each funcion will be used on a layer
        normalizers: Optional[List[Union[Normalizers, str]]] = None,
        data_path: Optional[str] = 'model_data.json'
    ) -> None:
        
        self.data = {}
        self.network = NeuralNetwork(
            shape=shape,
            activators=activators,
            normalizers=normalizers
        )
        self.network.inject_layers()
        
        with open(data_path, 'w') as file:
            file.seek(0)        
            json.dump(self.data, file)
        
    def save_data(
        self: 'Model',
        data_path: Optional[str] = 'model_data.json'
    ) -> None:
    
        weights = []
        biases = []
        for layer_weights, layer_biases in zip(*self.network.get_layers_parameters()):
            weights.append(layer_weights.tolist())
            biases.append(layer_biases.tolist())
        
        self.data = {
            #  Architecture!
            'shape': self.network.shape,
            'activators': self.network.activators,
            'normalizers': self.network.normalizers,
            
            #  Metrics!
            'score': self.score,
            'acuraccy': self.acuraccy,
            'epoch_count': self.epoch_count, 
            'input_count': self.input_count,
            'best_score': self.best_score,
            'best_acuraccy': self.best_acuraccy,
            
            #  Histories!
            'score_history': self.score_history,
            'accuracy_history': self.accuracy_history,

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
        self.epoch_count = self.data['epoch_count']
        self.input_count = self.data['input_count']
        self.best_score = self.data['best_score']
        self.best_acuraccy = self.data['best_acuraccy']
        
        #  Histories!
        self.score_history = self.data['score_history']
        self.accuracy_history = self.data['accuracy_history']
        
        self.network = NeuralNetwork(
            shape=self.data['shape'],
            activators=self.data['activators'],
            normalizers=self.data['normalizers']
        )
        self.network.inject_layers(
            weights=[np.array(x) for x in self.data['weights']],
            biases=[np.array(x) for x in self.data['biases']]
        )
        
    def training(
        self: 'Model',
        inputs: np.ndarray,
        target: Any,
        output_rule: Callable[[Any], Any],
        one_hot_vector: np.ndarray,
    )  -> np.ndarray:
        
        outputs = self.network.foward_propagation(inputs)
        self.network.backward_propagation(one_hot_vector)
        self.network.update_layers()
    
        output = output_rule(outputs)
        
        self.input_count += 1
        self._acuraccy_count += 1
        if target == output:
            self.score += 1
            self._hit_count += 1
        else: 
            self.score -= 1
        
        self.score_history.append(self.score)
        
        if self._acuraccy_count % 1000 == 0:
            self.acuraccy = float(f'{(self._hit_count / 1000) * 100:.3f}')           
            self.accuracy_history.append(self.acuraccy) 
            self._acuraccy_count = 0
            self._hit_count = 0
        
        if self.acuraccy > self.best_acuraccy[1]:
            self.best_acuraccy = (self.epoch_count, self.acuraccy)       
        if self.score > self.best_score[1]:
            self.best_score = (self.epoch_count, self.score)
        
        self.epoch_count += 1        
#:)