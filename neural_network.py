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
        self.shape = []
        self.layers = []
        
        #  Metrics!      
        self.total_epochs = 0
        self.loss = 1 
        self.score = 0
        self.acuraccy = 0
        self.best_loss = (0, 0) #  (Epoch, Loss)
        self.best_score = (0, 0) #  (Epoch, Score)
        self.best_acuraccy = (0, 0) #  (Epoch, Acuraccy)
        
        #  Histories!        
        self.loss_history = []
        self.score_history = []
        self.accuracy_history = []
        
        #  Private!
        self._grad_out = None
        self._hit_count = 0
        self._count_to_set_accuracy = 0
    
    def create(
        self: 'NeuralNetwork',
        layers: List[Layer]
    ) -> None:
        
        self.shape = []
        self.layers = layers

        for layer in self.layers:
            self.shape.append((layer.n_inputs, layer.n_neurons))

    def reset_metrics(
        self: 'NeuralNetwork'
    ) -> None:
        
        self.total_epochs = 0
        self.loss = 1 
        self.score = 0
        self.acuraccy = 0
        self.best_loss = (0, 0)
        self.best_score = (0, 0)
        self.best_acuraccy = (0, 0)

        self.loss_history = []
        self.score_history = []
        self.accuracy_history = []

    def save_data(
        self: 'NeuralNetwork',
        data_path: Optional[str] = 'model_data.json'
    ) -> None:
        
        data = {
            #  Architecture!
            'shape': self.shape,
            'activators': [layer.activator.__qualname__ for layer in self.layers],
            'activators_parameters': [layer.activator_parameters for layer in self.layers],

            #  Metrics!
            'total_epochs': self.total_epochs,
            'loss': self.loss,
            'score': self.score,
            'acuraccy': self.acuraccy,
            'best_loss': self.best_loss, 
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
            json.dump(data, file)
    
    def load_data(
        self: 'NeuralNetwork',
        data_path: Optional[str] = 'model_data.json'
    ) -> None:
        
        with open(data_path, 'r') as file:
            data = json.load(file)

        #  Architecture!
        self.shape = data['shape']

        for weights, biases, activator, activator_parameters in zip(
            data['weights'], data['biases'], data['activators'], data['activators_parameters']):
    
            self.layers.append(
                Layer(
                    weights=weights,
                    biases=biases,
                    activator=getattr(
                        globals()[activator.split('.')[0]], #  Getting the class name!
                        activator.split('.')[1]             #  Getting the function name!
                    ),
                    activator_parameters=activator_parameters
                )    
            )

        self.create(layers=self.layers)
        
        #  Metrics!
        self.total_epochs = data['total_epochs']
        self.loss = data['loss']
        self.score = data['score']
        self.acuraccy = data['acuraccy']
        self.best_loss = data['best_loss']
        self.best_score = data['best_score']
        self.best_acuraccy = data['best_acuraccy']
        
        #  Histories!
        self.score_history = data['score_history']
        self.accuracy_history = data['accuracy_history']
            
    def update_layers(
        self: 'NeuralNetwork',
        learning_rate: float
    ) -> None:
        
        for layer in self.layers:
            layer.update(learning_rate)
        
    def foward_propagation(
        self: 'NeuralNetwork',
        inputs: np.ndarray
    ) -> np.ndarray:

        self.inputs = inputs
        self.outputs = inputs
        for layer in self.layers:
            self.outputs = layer.foward(self.outputs)
            self.outputs = layer.activation(self.outputs)

        return self.outputs               

    def backward_propagation(
        self: 'NeuralNetwork',
        one_hot: np.ndarray
    ) -> None:
        
        reversed_layers = list(reversed(self.layers))
        for layer in reversed_layers:
            if self._grad_out is None:
                self._grad_out = self.outputs
           
            self._grad_out = layer.backward(
                one_hot=one_hot,
                pred_outputs=self._grad_out
            )
        self._grad_out = None
    
    def _metric_count(
        self: 'NeuralNetwork',
        target: Any,
        output: Any,
        one_hot: np.ndarray,
        pred_outputs: np.ndarray
    ) -> None:
        
        self.total_epochs += 1
        self._count_to_set_accuracy += 1
        
        self.loss = LossCalculations.categorical_crossentropy(
            one_hot_vector=one_hot,
            predicted_output=pred_outputs
        )

        if target == output:
            self.score += 1
            self._hit_count += 1
        else: 
            self.score -= 1        

        if self._count_to_set_accuracy % 1000 == 0:
            self.acuraccy = self._hit_count / 1000           
            self._hit_count = 0
            self._count_to_set_accuracy = 0

        self.loss_history.append(self.loss)
        self.score_history.append(self.score)
        self.accuracy_history.append(self.acuraccy)

        if self.loss < self.best_loss[1]:
            self.best_loss = (self.total_epochs, self.loss)
        if self.score > self.best_score[1]:
            self.best_score = (self.total_epochs, self.score)   
        if self.acuraccy > self.best_acuraccy[1]:
            self.best_acuraccy = (self.total_epochs, self.acuraccy)       
        
    def fit(
        self: 'NeuralNetwork',
        inputs: List[int],
        target: Any,
        one_hot: np.ndarray,
        output_rule: Callable[[Any], Any],
        learning_rate: float
    ) -> None: 
        
        outputs = self.foward_propagation(np.array(inputs, dtype=np.float64))
        self.backward_propagation(one_hot)
        self.update_layers(learning_rate)
        output = output_rule(outputs)
            
        self._metric_count(
            target=target,
            output=output,
            one_hot=one_hot,
            pred_outputs=outputs
        )
    
    def predict(
        self: 'NeuralNetwork',
        inputs: List[int],
        target: Any,
        one_hot: np.ndarray,
        output_rule: Callable[[Any], Any]
    ) -> np.ndarray:
        
        outputs = self.foward_propagation(np.array(inputs, dtype=np.float64))
        output = output_rule(outputs)

        self._metric_count(
            target=target,
            output=output,
            one_hot=one_hot,
            pred_outputs=outputs
        )

        return outputs, output
#:)