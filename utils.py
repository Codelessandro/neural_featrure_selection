import numpy as np


def to_row(datasets):

    _datasets=[]
    for dataset in datasets:
        for column in  dataset[1]:
            flattened_dataset = dataset[0].flatten()
            column_values = column[0]
            score = column[1]
            _=np.concatenate((flattened_dataset, column_values))
            _datasets.append(np.append(_,score))

    return np.array(_datasets)