from neural_network import NeuralNetwork, Layer
from activation_module import ActivationFunctions
from fake_fruits import FruitsData, FakeFruit
import numpy as np



n_epochs = 100
score = 0

for epoch in range(1, n_epochs + 1):
    
    model = NeuralNetwork()

    model.create(
    net_shape=[(3, 5), (5, 5)],
    activators=[ActivationFunctions.softmax, ActivationFunctions.softmax]
    )

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

print(score)
