from config import *
import numpy as np
import math
import pdb

class DummyData():

    def __init__(self,columns,rows,task):
        self.columns = columns
        self.rows=rows
        self.task=task
        pass

    def get_data(self):
        return np.zeros(shape=(self.columns,self.rows))

    def get_add_column(self):
        column = np.zeros( shape=(self.rows ))
        score = math.ceil( np.random.uniform(0,1,1)*10)
        return column,score


def gen_dummy_data_sets():
    datasets = []
    for i in np.arange(config["nr_dummy_datasets"]):
        dataset = DummyData(4,4,"Classification")
        data = dataset.get_data()
        columns=[]
        for i in np.arange(config["nr_additional_columns_per_dataset"]):
            add_columns = dataset.get_add_column()
            columns.append(add_columns)
        datasets.append([data,columns])

    return to_row(datasets)

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