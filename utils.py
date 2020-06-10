import numpy as np
from config import *
import pdb

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


def batchify(base_x,  add_columns, base_y, budget):
    x=[]
    y_data=[]
    y_score=[]
    extended_dataset = []
    if budget == 1:
        for add_column in add_columns:
            extended_dataset = np.hstack((base_x, np.expand_dims(add_column[0],axis=1)))
            x_batch = generate_batch(extended_dataset)
            y_data_batch = generate_batch(base_y)
            y_score_batch = np.ones(x_batch.shape[0]) * add_column[1]
            x.append(x_batch)
            y_data.append(y_data_batch)
            y_score.append(y_score_batch)
        x=np.concatenate(x,axis=0 )
        y_score=np.concatenate(y_score,axis=0)
        y_data=np.concatenate(y_data,axis=0)
    else:
        budget_columns = np.zeros(shape=[len([row[0] for row in add_columns][0]), 1])
        add_columns = add_columns[: budget]
        for add_column in add_columns:
            budget_columns = np.hstack((budget_columns, np.expand_dims(add_column[0].T, axis=1)))
        #budget_columns = budget_columns[:][1:]
        extended_dataset = np.hstack((base_x, budget_columns))
        x = generate_batch(extended_dataset)
        y_data = generate_batch(base_y)
        for add_column in add_columns:
            y_score = np.ones(x.shape[0]) * add_column[1]
        #x.append(x_batch)
        #y_data.append(y_data_batch)
        #y_score.append(y_score_batch)
    #pdb.set_trace()
    return x, y_data, y_score  #x of shape: nr_batch, batch_size, base_columns + one add_column


def normalize(data):
    normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    return np.tanh(normalized*100)
