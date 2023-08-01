import os
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from fake_fruits import FruitsData, FakeFruit
from activation_module import ActivationFunctions

score = 0
n_epochs = 1000000
layer_shape = [(3, 25), (25, 25), (25, 25)]
x_coord = []
y_coord = []

model = NeuralNetwork()

model.create(
    net_shape=layer_shape,
    activators=[ActivationFunctions.softmax, ActivationFunctions.softmax, ActivationFunctions.softmax]
    )

for epoch in range(1, n_epochs + 1):

    fruit = FakeFruit()
    inputs = np.array(
        object=[fruit.diameter, fruit.weight, fruit.texture],
        dtype=np.float64
    )

    outputs = model.foward_propagation(inputs)
    model.backward_propagation(fruit.one_hot_vector)
    predicted_fruit = FruitsData.fruits_data[np.argmax(outputs)][0]

    if fruit.name == predicted_fruit:
        score += 1
    else:
        score -= 1

    x_coord.append(epoch)
    y_coord.append(score)
    
    os.system('cls')
    print('-'*100, 
          f'\nScore: {score}',
          f'\nProgress: {epoch / n_epochs * 100:.3f}%',
          f'\nRemaining epochs: {n_epochs + 1 - epoch}'
    )

model.save_data()
plt.plot(
    x_coord,
    y_coord,
    color='Red'
)
plt.title(f'Using Softmax\n{layer_shape}')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.show()
input()