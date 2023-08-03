import os
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from fake_fruits import FruitsData, FakeFruit
from activation_module import ActivationFunctions

n_epochs = 1000
layer_shape = [(5, 25), (25, 25)]


score = 0
correct_predictions = 0
one_hundred_epochs = 0
accuracy = 0

epochs_x_coord = []
one_hundred_epochs_x_coord = []
score_y_coord = []
accuracy_y_coord = []


model = NeuralNetwork()

model.create(
    net_shape=layer_shape,
    activators=[ActivationFunctions.softmax, ActivationFunctions.softmax]
    )

for epoch in range(1, n_epochs + 1):

    fruit = FakeFruit()
    inputs = np.array(
        object=[fruit.diameter,
                fruit.weight,
                fruit.texture,
                fruit.sugar_level,
                fruit.ph_level],
        dtype=np.float64
    )

    outputs = model.foward_propagation(inputs)
    model.backward_propagation(fruit.one_hot_vector)
    predicted_fruit = FruitsData.fruits_data[np.argmax(outputs)][0]

    if fruit.name == predicted_fruit:
        score += 1
        correct_predictions += 1
    else:
        score -= 1

    if epoch % 100 == 1:
        one_hundred_epochs += 1
        accuracy = (correct_predictions / epoch) * 100 
        correct_predictions = 0
        
    epochs_x_coord.append(epoch)
    one_hundred_epochs_x_coord.append(one_hundred_epochs)
    score_y_coord.append(score)
    accuracy_y_coord.append(accuracy)        
    
    if epoch / 100 == 1:
        model.save_data()

    os.system('cls')
    print('-'*100, 
          f'\nScore: {score}',
          f'\nAccuracy: {accuracy:.3f}',
          f'\nProgress: {epoch / n_epochs * 100:.3f}%',
          f'\nRemaining epochs: {n_epochs - epoch}'
    )

model.save_data()

plt.plot(
    one_hundred_epochs_x_coord,
    accuracy_y_coord,
    color='Blue'
)
plt.title(f'Using Softmax\n{layer_shape}')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.show()
input()