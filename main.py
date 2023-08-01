from neural_network import NeuralNetwork, Layer
from activation_module import ActivationFunctions
from fake_fruits import FruitsData, FakeFruit
import numpy as np

model = NeuralNetwork()
model.create(
    net_shape=[(3, 5), (5, 5)],
    activators=[ActivationFunctions.softmax, ActivationFunctions.softmax]
)

n_epochs = 1
score = 0

for epoch in range(1, n_epochs + 1):
    
    fruit = FakeFruit()
    inputs = np.array(
        object=[fruit.diameter, fruit.weight, fruit.texture],
        dtype=np.float64
    )
    
    a = Layer(
        activator=ActivationFunctions.softmax,
        weights=0.10 * np.random.randn(3, 5),
        biases=np.zeros((1, 5))
    )
    b = Layer(
        activator=ActivationFunctions.softmax,
        weights=0.10 * np.random.randn(5, 5),
        biases=np.zeros((1, 5))
    )
    a_outputs = a.foward(inputs)
    b_outputs = a.foward(a_outputs)

    """outputs = model.foward_propagation(inputs)
    model.backward_propagation(fruit.one_hot_vector)

    output_fruit = FruitsData.fruits_data[
        np.argmax(outputs)
    ]"""

    print(b_outputs)