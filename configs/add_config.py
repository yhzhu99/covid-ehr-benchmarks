configs = {
    'depth': [3, 6, 9],
    'iterations': [50, 100, 150],
    'learning_rate': [0.1, 0.5, 1]
}

dataset_name = {
    'hm': 'hm',
    'tj': 'tongji'
}

model_name = 'catboost'

for data in ['tj', 'hm']:
    for task in ['outcome', 'los']:
        for md in configs['depth']:
            for iter in configs['iterations']:
                for lr in configs['learning_rate']:
                    with open(f'{data}_{task}_{model_name}_kf10_md{md}_iter{iter}_lr{lr}.yaml', mode='w') as file:
                        file.write(
                            f'name: {data}_{task}_{model_name}_kf10_md{md}_iter{iter}_lr{lr}\n' +
                            f'model_type: ml\n' +
                            f'model: {model_name}\n' +
                            f'mode: val\n' +
                            f'dataset: {dataset_name[data]}\n' +
                            f'task: {task}\n' +
                            f'depth: {md}\n' +
                            f'iterations: {iter}\n' +
                            f'learning_rate: {lr}\n'
                        )

# for data in ['tj', 'hm']:
#     for task in ['outcome', 'los']:
#         for md in configs['depth']:
#             with open(f'{data}_{task}_{model_name}_kf10_md{md}.yaml', mode='w') as file:
#                 file.write(
#                     f'name: {data}_{task}_{model_name}_kf10_md{md}\n' +
#                     f'model_type: ml\n' +
#                     f'model: {model_name}\n' +
#                     f'mode: val\n' +
#                     f'dataset: {dataset_name[data]}\n' +
#                     f'task: {task}\n' +
#                     f'depth: {md}\n'
#                 )