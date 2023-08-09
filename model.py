import json
import numpy as np
import threading
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
        self.total_inputs = 0
        self.best_score = (0, 0) #  (Epoch, Score)
        self.best_acuraccy = (0, 0) #  (Epoch, Acuraccy)
        
        #  Histories!        
        self.score_history = []
        self.accuracy_history = []
        
        #  Private!
        self._hit_count = 0
        self._input_count = 0
        self._score_count = 0
        self._batch_grads = []
        self._batch_threads = []
        
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
            'total_epochs': self.total_epochs, 
            'total_inputs': self.total_inputs,
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
        self.total_epochs = self.data['total_epochs']
        self.total_inputs = self.data['total_inputs']
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
    
    def _get_metric(
        self: 'Model'
    ) -> None:
        
        self.score_history.append(self.score)
        if self._input_count % 100 == 0:
            self.score += float(f'{(self._score_count / 100):.3f}')
            self.acuraccy = float(f'{(self._hit_count / 100) * 100:.3f}')           
            
            self.score_history.append(self.score)
            self.accuracy_history.append(self.acuraccy)
             
            self._input_count = 0
            self._score_count = 0
            self._hit_count = 0
            
        if self.acuraccy > self.best_acuraccy[1]:
            self.best_acuraccy = (self.total_epochs, self.acuraccy)       
        if self.score > self.best_score[1]:
            self.best_score = (self.total_epochs, self.score)
        
    def _metric_count(
        self: 'Model',
        target: Any,
        output: Any
    ) -> None:
        
        self.total_inputs += 1
        self._input_count += 1
        
        if target == output:
            self._score_count += 1
            self._hit_count += 1
        else: 
            self._score_count -= 1        

        
    def _get(
        self: 'Model',
        inputs: np.ndarray,
        target: Any,
        output_rule: Callable[[Any], Any],
        one_hot_vector: np.ndarray
    ) -> None:
        
        private_network = self.network
        outputs = private_network.foward_propagation(inputs)
        private_network.backward_propagation(one_hot_vector)    
        output = output_rule(outputs)
    
        self._batch_grads.append(private_network.get_gradients())
       
        
    def batch_training(
        self: 'Model',
        all_inputs: np.ndarray,
        all_targets: Any,
        output_rule: Callable[[Any], Any],
        all_one_hot_vectors: np.ndarray
    )  -> np.ndarray:
    
        self._batch_grads.clear()
        self._batch_threads.clear()
        for inputs, target, one_hot_vector in zip(
            all_inputs, all_targets, all_one_hot_vectors
        ):
            thread = threading.Thread(
                target=self._get, 
                args=(inputs, target, output_rule, one_hot_vector)
            )
            thread.start()
            self._batch_threads.append(thread)
        
        batch_grad_weights = [x[0] for x in self._batch_grads]
        batch_grad_biases = [x[1] for x in self._batch_grads]             
        
        
        for thread in self._batch_threads:
            thread.join()   
        
        self.total_epochs += 1        
#:)