import numpy as np
from config import *


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

def generate_batch(extended_dataset):
    cutoff=extended_dataset.shape[0]%config["batch_size"]
    extended_dataset=extended_dataset[0:extended_dataset.shape[0]-cutoff]
    return extended_dataset.reshape( -1, config["batch_size"] , extended_dataset.shape[1])

def batchify(datasets):
    dataset = datasets[0] #we only one dataset at this point
    x=[]
    y=[]
    for add_column in dataset[1]:
        extended_dataset = np.hstack((dataset[0], np.expand_dims(add_column[0],axis=1)))
        x_batch = generate_batch(extended_dataset)
        y_batch = np.ones(x_batch.shape[0]) * add_column[1]

        x.append(x_batch)
        y.append(y_batch)

    x=np.concatenate(x,axis=0 )
    y=np.concatenate(y,axis=0 )

    return x, y #x of shape: nr_batch, batch_size, base_columns + one add_column


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))