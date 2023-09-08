import os
import numpy as np
import matplotlib.pyplot as plt
from fake_fruits import FakeFruit, FruitsData
from neural_network import NeuralNetwork
from layer import Layer
from utils import *

model = NeuralNetwork()

if not os.path.exists('model_data.json'):
    model.create([
        Layer(
            weights=Initializers.random_weights((5, 25)),
            biases=Initializers.zeros_biases((5, 25)),
            activator=ActivationFunctions.softmax,
            normalizer=Normalizers.minmax
        ),
        Layer(
            weights=Initializers.random_weights((25, 25)),
            biases=Initializers.zeros_biases((25, 25)),
            activator=ActivationFunctions.softmax
        ),
        Layer(
            weights=Initializers.random_weights((25, 25)),
            biases=Initializers.zeros_biases((25, 25)),
            activator=ActivationFunctions.softmax
        )
    ])
else:
    model.load_data()


n_epochs = int(input('n_epochs: ')) + 1
for epoch in range(1, n_epochs):
    
    fruit = FakeFruit()
    model.learn(
        inputs=[
            fruit.weight,
            fruit.texture,
            fruit.diameter, 
            fruit.ph_level, 
            fruit.sugar_level
        ],
        target=fruit.name,
        output_rule=lambda x:FruitsData.fruits_data[np.argmax(x)][0],
        one_hot_vector=fruit.one_hot_vector,
        epoch=epoch
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
fig, axs = plt.subplots(1, 2, figsize=(12, 5))  
fig.suptitle(f'Training', fontsize=20)

axs[0].plot(
    [x * 1000 for x in range(len(model.accuracy_history))],
    model.accuracy_history,
    color='Blue'
)
axs[0].grid(True)
axs[0].set_title(f'Acuraccy\nBest: {model.best_acuraccy[1]}', fontsize=16)
axs[0].set_xlabel('Epoch', fontsize=14)

axs[1].plot(
    [x for x in range(model.total_epochs)],
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