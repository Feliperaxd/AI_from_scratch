import json
import numpy as np
from utils import *
from layer import Layer
from typing import Any, Callable, List, Optional


class NeuralNetwork:


    def __init__(
        self: 'NeuralNetwork'
    ) -> None: 

        self.inputs = None
        self.outputs = None
        
        #  Architecture!
        self.shape = None
        self.layers = None
        self.activators = None
        self.normalizers = None

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
        self._grad_outputs = None
        self._hit_count = 0
        self._count_to_set_accuracy = 0
    
    def create(
        self: 'NeuralNetwork',
        layers: List[Layer]
    ) -> None:
        
        self.shape = []
        self.layers = layers
        self.activators = []
        self.normalizers = []

        for layer in self.layers:
            self.shape.append((layer.n_inputs, layer.n_neurons))
            self.activators.append(layer.activator)
            self.normalizers.append(layer.normalizer)

    def save_data(
        self: 'NeuralNetwork',
        data_path: Optional[str] = 'model_data.json'
    ) -> None:
        
        self.data = {
            #  Architecture!
            'shape': self.shape,
            'activators': [x.__qualname__ for x in self.activators],
            'normalizers': [x.__qualname__ for x in self.normalizers if x is not None],
            
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
            'weights': [x.weights.tolist() for x in self.layers],
            'biases': [x.biases.tolist() for x in self.layers]
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

        #  Architecture!
        self.shape = self.data['shape']
        self.layers = []

        for weights, biases, activator, normalizer in zip(
            self.data['weights'], self.data['biases'], self.data['activators'], self.data['normalizers']):
            self.layers.append(
                Layer(
                    weights=weights,
                    biases=biases,
                    activator=getattr(
                        globals()[activator.split('.')[0]], #  Getting the class reference!
                        activator.split('.')[1]             #  Getting the function reference!
                    ),
                    normalizer=getattr(
                        globals()[normalizer.split('.')[0]], 
                        normalizer.split('.')[1]
                    )
                )    
            )
        self.create(layers=self.layers)

        #  Metrics!
        self.score = self.data['score']
        self.acuraccy = self.data['acuraccy']
        self.total_epochs = self.data['total_epochs']
        self.best_score = self.data['best_score']
        self.best_acuraccy = self.data['best_acuraccy']
        
        #  Histories!
        self.score_history = self.data['score_history']
        self.accuracy_history = self.data['accuracy_history']
            
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
        one_hot_vector: np.ndarray,
    ) -> None:
        
        reversed_layers = list(reversed(self.layers))
        for layer in reversed_layers:
            if self._grad_outputs is None:
                self._grad_outputs = self.outputs
           
            self._grad_outputs = layer.backward(
                pred_outputs=self._grad_outputs,
                one_hot_vector=one_hot_vector
            )
        self._grad_outputs = None
    
    def _metric_count(
        self: 'NeuralNetwork',
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
        self: 'NeuralNetwork',
        inputs: List[int],
        target: Any,
        output_rule: Callable[[Any], Any],
        one_hot_vector: np.ndarray
    ) -> None: 
        
        outputs = self.foward_propagation(np.array(inputs, dtype=np.float64))
        self.backward_propagation(one_hot_vector)
        self.update_layers()
        output = output_rule(outputs)
            
        self._metric_count(
            target=target,
            output=output
        )
#:)