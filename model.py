import json
import numpy as np
from neural_network import NeuralNetwork
from activation_module import ActivationFunctions
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
        self.last_epoch = 0
        self.best_score = (0, 0) #  (Epoch, Score)
        self.best_acuraccy = (0, 0) #  (Epoch, Acuraccy)
        self.training_count = 0

        #  Histories!        
        self.score_history = []
        self.accuracy_history = []
        
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
            'last_epoch': self.last_epoch, 
            'best_score': self.best_score,
            'best_acuraccy': self.best_acuraccy,
            'training_count': self.training_count,
            
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
        self.last_epoch = self.data['last_epoch']
        self.best_score = self.data['best_score']
        self.best_acuraccy = self.data['best_acuraccy']
        self.training_count = self.data['training_count']
        
        #  Histories!
        self.score_history = self.data['score_history']
        self.accuracy_history = self.data['accuracy_history']
        
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
        target: Any,
        inputs: List[float],
        output_rule: Callable[[Any], Any],
        one_hot_vector: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        
        outputs = self.network.foward_propagation(inputs)
        delta_outputs = self.network.backward_propagation(one_hot_vector)
        output = output_rule(outputs)

        self.training_count += 1
        self.last_epoch += 1
        self._epoch_count += 1    
        self.score_history.append(self.score)

        if target == output:
            self.score += 1
            self._hit_count += 1
        else: 
            self.score -= 1
          
        if self._epoch_count % 1000 == 0:
            self.acuraccy = float(f'{(self._hit_count / 1000) * 100:.3f}')           
            self.accuracy_history.append(self.acuraccy) 
            self._epoch_count = 0
            self._hit_count = 0
        
        if self.acuraccy > self.best_acuraccy[1]:
            self.best_acuraccy = (self.last_epoch, self.acuraccy)       
        if self.score > self.best_score[1]:
            self.best_score = (self.last_epoch, self.score)
        
        return outputs, delta_outputs, output

    def operate(
        self: 'Model',
        inputs: np.ndarray,
        one_hot_vector: np.ndarray
    ) -> np.ndarray:
        
        outputs = self.foward_propagation(inputs)
        self.backward_propagation(one_hot_vector)
        
        return outputs
#:)