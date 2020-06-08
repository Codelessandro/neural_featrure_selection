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

def generate_batch(dataset):
    cutoff=dataset.shape[0]%config["batch_size"]
    dataset=dataset[0:dataset.shape[0]-cutoff]
    if len(dataset.shape)==1:
        return  dataset.reshape(-1,config["batch_size"])
    else:
        return dataset.reshape(-1, config["batch_size"], dataset.shape[1])


def batchify(base_x,  add_columns, base_y):
    x=[]
    y_data=[]
    y_score=[]
    for add_column in add_columns:
        extended_dataset = np.hstack((base_x, np.expand_dims(add_column[0],axis=1)))
        x_batch = generate_batch(extended_dataset)
        y_data_batch = generate_batch(base_y)
        y_score_batch = np.ones(x_batch.shape[0]) * add_column[1]

        x.append(x_batch)
        y_data.append(y_data_batch)
        y_score.append(y_score_batch)

    x=np.concatenate(x,axis=0 )
    y_score=np.concatenate(y_score,axis=0 )
    y_data=np.concatenate(y_data,axis=0 )

    return x, y_data, y_score  #x of shape: nr_batch, batch_size, base_columns + one add_column


def normalize(data):
    normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    return np.tanh(normalized*100)