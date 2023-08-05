from model import Model


model = Model()
model.create(
    [(5, 25), (25, 25), (25, 25)], 
    ['softmax', 'softmax', 'softmax']
    )

print(f'''
        score: {model.score}
        acuraccy: {model.acuraccy}
        total_epochs: {model.total_epochs}
        better_score: {model.better_score}
        better_acuraccy: {model.better_acuraccy}
        training_count: {model.training_count}
        '''
    )