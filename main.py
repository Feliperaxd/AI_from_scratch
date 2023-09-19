import os
import numpy as np
import matplotlib.pyplot as plt
from fake_fruits import FakeFruit, FruitsData
from neural_network import NeuralNetwork
from layer import Layer
from utils import *
from keras.datasets import mnist 

model = NeuralNetwork()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

n_epochs = int(input('n_epochs: ')) + 1


if not os.path.exists('model_data.json'):
    model.create([
        Layer(
            weights=Initializers.random_weights((784, 50)),
            biases=Initializers.zeros_biases((784, 50)),
            activator=ActivationFunctions.softmax,
        ),
        Layer(
            weights=Initializers.random_weights((50, 10)),
            biases=Initializers.zeros_biases((50, 10)),
            activator=ActivationFunctions.softmax,
        )
    ])
else:
    model.load_data()
    
for epoch in range(1, n_epochs):
    
    model.fit(
        inputs=x_train[epoch].flatten() / 255,
        target=y_train[epoch],
        one_hot=[1 if x == y_train[epoch] else 0 for x in range(10)],
        output_rule=lambda x: np.argmax(x),
        learning_rate=0.01
        )

    """os.system('cls')
    print(f'''
            ---Progress {(epoch / n_epochs) * 100:.2f}%---
            score: {model.score}
            acuraccy: {model.acuraccy}%
            total_epochs: {model.total_epochs}
            best_score: {model.best_score[1]}
            best_acuraccy: {model.best_acuraccy[1]}%
            '''
        )"""

model.save_data()
epoch_history = [x for x in range(model.total_epochs)]

fig, axs = plt.subplots(1, 2, figsize=(12, 5))  
fig.suptitle(f'Training', fontsize=20)

axs[0].plot(
    epoch_history,
    model.accuracy_history,
    color='Blue'
)
axs[0].grid(True)
axs[0].set_title(f'Acuraccy\nBest: {model.best_acuraccy[1]}', fontsize=16)
axs[0].set_xlabel('Epoch', fontsize=14)

axs[1].plot(
    epoch_history,
    model.score_history,
    color='Red'
)
axs[1].grid(True)
axs[1].set_title(f'Score\nBest: {model.best_score[1]}', fontsize=16)
axs[1].set_xlabel('Epoch', fontsize=14)


plt.tight_layout()
plt.show()
input()
#:)