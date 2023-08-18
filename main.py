import os
import threading
import numpy as np
from model import Model
import matplotlib.pyplot as plt
from fake_fruits import FakeFruit, FruitsData


model = Model()

if not os.path.exists('model_data.json'):
    model.create(
        [(5, 25), (25, 25), (25, 25), (25, 25)], 
        ['leaky_relu', 'leaky_relu', 'leaky_relu',  'softmax'],
        ['minmax', 'minmax', 'minmax']
        )
else:
    model.load_data()

n_epochs = int(input('n_epochs: ')) + 1
for epoch in range(1, n_epochs):
    
    fruit = FakeFruit()
    model.learn(
        inputs=np.array([
            fruit.weight,
            fruit.texture,
            fruit.diameter, 
            fruit.ph_level, 
            fruit.sugar_level
            ]),
        target=fruit.name,
        output_rule=lambda x:FruitsData.fruits_data[np.argmax(x)][0],
        one_hot_vector=fruit.one_hot_vector
    )
    
    if epoch % 100000 == 0:
        model.save_data()
        model.save_data('backup.json')
        
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