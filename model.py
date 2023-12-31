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
        self.total_epochs = 0
        self.best_score = (0, 0) #  (Epoch, Score)
        self.best_acuraccy = (0, 0) #  (Epoch, Acuraccy)
        
        #  Histories!        
        self.score_history = []
        self.accuracy_history = []
        
        #  Private!
        self._threads = []
        self._hit_count = 0
        self._count_to_set_accuracy = 0
        
    def save_data(
        self: 'Model',
        data_path: Optional[str] = 'model_data.json'
    ) -> None:
        
        self.data = {
            #  Network Architecture!
            'shape': self.network.shape,
            'activators': self.network.activators,
            'normalizers': self.network.normalizers,
            
            #  Metrics!
            'score': self.score,
            'acuraccy': self.acuraccy,
            'total_epochs': self.total_epochs, 
            'best_score': self.best_score,
            'best_acuraccy': self.best_acuraccy,
            
            #  Histories!
            'score_history': self.score_history,
            'accuracy_history': self.accuracy_history,

            #  Tensors!
            'weights': [layer.weights for layer in self.network.layers],
            'biases': [layer.biases for layer in self.network.layers]
        }

        with open(data_path, 'w') as file:
            file.seek(0)        
            json.dump(self.data, file)
        
    def load_data(
        self: 'Model',
        data_path: Optional[str] = 'model_data.json'
    ) -> None:
        
        with open(data_path, 'r') as file:
            self.data = json.load(file)

        #  Architecture!    
    
        

        #  Metrics!
        self.score = self.data['score']
        self.acuraccy = self.data['acuraccy']
        self.total_epochs = self.data['total_epochs']
        self.best_score = self.data['best_score']
        self.best_acuraccy = self.data['best_acuraccy']
        
        #  Histories!
        self.score_history = self.data['score_history']
        self.accuracy_history = self.data['accuracy_history']
        
    def _metric_count(
        self: 'Model',
        target: Any,
        output: Any
    ) -> None:
        
        self.total_epochs += 1
        self._count_to_set_accuracy += 1
        
        if target == output:
            self.score += 1
            self._hit_count += 1
        else: 
            self.score -= 1        
        
        self.score_history.append(self.score)
        if self._count_to_set_accuracy % 1000 == 0:
            self.acuraccy = float(f'{(self._hit_count / 1000) * 100:.3f}')           
            self.accuracy_history.append(self.acuraccy)
            self._hit_count = 0
            self._count_to_set_accuracy = 0
            
        if self.acuraccy > self.best_acuraccy[1]:
            self.best_acuraccy = (self.total_epochs, self.acuraccy)       
        if self.score > self.best_score[1]:
            self.best_score = (self.total_epochs, self.score)   
    
    def learn(
        self: 'Model',
        inputs: List[int],
        target: Any,
        output_rule: Callable[[Any], Any],
        one_hot_vector: np.ndarray
    ) -> None: 
        
        outputs = self.network.foward_propagation(np.array(inputs))
        self.network.backward_propagation(one_hot_vector)
        self.network.update_layers(learning_rate=0.1)
        output = output_rule(outputs)
            
        self._metric_count(
            target=target,
            output=output
        )
#:)