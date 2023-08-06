import os
import numpy as np
from model import Model
import matplotlib.pyplot as plt
from fake_fruits import FakeFruit, FruitsData

model = Model()

if not os.path.exists('model_data.json'):
    model.create(
        [(5, 25), (25, 25), (25, 25)], 
        ['softmax', 'softmax', 'softmax']
        )
else:
    model.load_data()

for epoch in range(1000):

    fruit = FakeFruit()
    model.training(
        epoch=epoch,
        target=fruit.name,
        inputs=[
            fruit.diameter,
            fruit.weight,
            fruit.texture,
            fruit.ph_level,
            fruit.sugar_level
        ],
        output_rule=lambda x:FruitsData.fruits_data[np.argmax(x)][0],
        one_hot_vector=fruit.one_hot_vector
    )

    if epoch % 1000 == 1:
        model.save_data()

    os.system('cls')
    print(f'''
            score: {model.score}
            acuraccy: {model.acuraccy}
            total_epochs: {model.total_epochs}
            better_score: {model.better_score}
            better_acuraccy: {model.better_acuraccy}
            training_count: {model.training_count}
            '''
        )


model.save_data()
plt.title('Training')
plt.plot(
    [x for x in range(model.total_epochs)],
    model.score_coord,
    color='Red',
    label='Score'
)
plt.plot(
    [x for x in range(model.total_epochs)],
    model.accuracy_coord,
    color='Blue',
    label='Accuracy'
)
plt.legend(fontsize='medium')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.show()
#:)