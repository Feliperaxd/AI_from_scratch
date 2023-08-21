import json
import threading
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
        self._avg_weight_gradients = []
        self._avg_bias_gradients = []

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
    
    def _batch_foward_process(
        self: 'Model',
        inputs: List[int],
        target: List[Any],
        output_rule: Callable[[Any], Any],
        one_hot_vector: np.ndarray
    ) -> None: 
        
        outputs = self.network.foward_propagation(np.array(inputs))
        self.network.backward_propagation(one_hot_vector)    
        output = output_rule(outputs)    
        
        #  Add in the list to calculate the average!
        #  NOTE: This list must have previously a zero for each layer! 
        for layer_index, (gw, gb) in enumerate(zip(*self.network.get_layers_gradients())):
            self._avg_weight_gradients[layer_index] += gw
            self._avg_bias_gradients[layer_index] += gb
            
        self._metric_count(
            target=target,
            output=output
        )
        
    def batch_training(
        self: 'Model',
        all_inputs: List[List[int]],
        all_targets: List[Any],
        output_rule: Callable[[Any], Any],
        all_one_hot_vectors: List[np.ndarray]
    )  -> None:
        
        batch_size = len(all_inputs)
        
        #  Adding a zero per layer!
        self._avg_weight_gradients = [0 for _ in range(len(self.network.layers))]
        self._avg_bias_gradients = [0 for _ in range(len(self.network.layers))]

        self._threads.clear()

        for inputs, target, one_hot_vector in zip(all_inputs, all_targets, all_one_hot_vectors):
            thread = threading.Thread(
                target=self._batch_foward_process,
                args=(
                    inputs, target, 
                    output_rule, one_hot_vector
                )
            )
            self._threads.append(thread)
            thread.start()

        for thread in self._threads:
            thread.join()

        for layer_index in range(len(self.network.layers)):
            self._avg_weight_gradients[layer_index] /= batch_size
            self._avg_bias_gradients[layer_index] /= batch_size
        
        """self.network.update_layers(
            grad_weights=self._avg_weight_gradients,
            grad_biases=self._avg_bias_gradients
        )"""

    def teste(
        self: 'Model',
        inputs: np.ndarray,
        output_rule: Callable[[Any], Any],
    ) -> Any:
             
        pass
#:)