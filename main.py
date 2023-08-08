import os
import threading
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

n_epochs = int(input('n_epochs: ')) + 1
batch_size = int(input('batch_size: '))
def training():
    fruit = FakeFruit()
    model.training(
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
    

threads = []
for epoch in range(1, n_epochs):
    for batch in range(batch_size):
        thread = threading.Thread(target=training)
        threads.append(thread)
        thread.start()
    
    model.last_epoch += 1
    os.system('cls')
    print(f'''
            ---Progress {(epoch / n_epochs) * 100:.2f}%---
            score: {model.score}
            acuraccy: {model.acuraccy}%
            last_epoch: {model.last_epoch}
            best_score: {model.best_score[1]}
            best_acuraccy: {model.best_acuraccy[1]}%
            training_count: {model.training_count}
            '''
        )
    USAR BATCH DO JEITO CORRETO
    for thread in threads:
        thread.join()

model.save_data()
fig, axs = plt.subplots(1, 2, figsize=(12, 5))  
fig.suptitle(f'Training', fontsize=20)

axs[0].plot(
    [x * 1000 for x in range(len(model.accuracy_history))],
    model.accuracy_history,
    color='Blue'
)
axs[0].set_title(f'Acuraccy\nBest: {model.best_acuraccy[1]}')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Metrics')

axs[1].plot(
    [x for x in range(model.training_count)],
    model.score_history,
    color='Red'
)
axs[1].set_title(f'Score\nBest: {model.best_score[1]}')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Metrics')

plt.tight_layout()
plt.show()
input()
#:)