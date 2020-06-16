import pdb
import pandas as pd
from numpy import genfromtxt
from sklearn import datasets, linear_model
import numpy as np
from utils import *
from numpy.random import default_rng

np.random.seed(0)

from config import *

class BaseData():
    def __init__(self, dataset_path, delimiter, target, combine, base_size=5, rifs=False):

        self.augmented_nr_rows = config["dataset_rows"]
        self.dataset_path = dataset_path
        self.delimiter = delimiter
        self.target = target
        self.combine = combine
        self.base_size = base_size
        self.rifs = rifs


    def regression_score(self, x,y):
        regr = linear_model.LinearRegression()
        regr.fit(x,y)
        return regr.score(x,y)

    def load(self):
        self.dataset = genfromtxt(self.dataset_path, delimiter=self.delimiter, encoding="utf8", invalid_raise = False)[1:]
        self.dataset = np.nan_to_num(self.dataset)
        self.base_dataset = self.dataset
        self.combination(self.dataset, self.target, self.combine)

    def combination(self, dataset, target, combine):
        self.y_data = np.empty(shape=[1, config['batch_size']])

        if config["budget_join"]:
            self.x = np.empty(shape=[1, config['batch_size'], config['nr_base_columns']+ config['nr_add_columns_per_budget_group'] ])
            self.y_score = np.empty(shape=[1,config['nr_add_columns_per_budget_group'],])
        else:
            self.x = np.empty(shape=[1, config['batch_size'], config['nr_base_columns']+1])
            self.y_score = np.empty(shape=[1])


        for _ in range(combine):
            random_col = np.random.choice(np.delete(np.arange(dataset.shape[1]),target),size=self.base_size,replace=False)
            x, y_data, y_score = self.generate_data(random_col, self.target)

            self.x = np.concatenate((self.x, x))

            try:
                self.y_data = np.concatenate((self.y_data, y_data))
                self.y_score = np.concatenate((self.y_score, y_score))
            except:
                import pdb; pdb.set_trace()


        self.x = self.x[1:]
        self.y_data = self.y_data[1:]
        self.y_score = self.y_score[1:]
        #pdb.set_trace()
        self.xy = np.concatenate((self.x, np.expand_dims(self.y_data, axis=2)), axis=2)

    def generate_data(self, base_dependent_columns, independent_column):
        base_x = self.dataset[:, base_dependent_columns]
        self.base_x = base_x
        base_y = self.dataset[:, independent_column]
        self.base_y = base_y
        self.base_xy = np.concatenate((self.base_x, np.expand_dims(self.base_y, axis=1)), axis=1)

        if self.rifs is True:
            random_column = np.random.choice(int(np.amax(self.dataset)), size=self.dataset.shape[0], replace=True)
            rifs_base_x = np.concatenate((base_x, random_column[:, None]), axis=1)
            base_r2_score = self.regression_score(rifs_base_x, base_y)
        else:
            try:
                base_r2_score = self.regression_score(base_x, base_y)
            except:
                pdb.set_trace()

        add_columns = []
        size = self.dataset.shape[1] - len(base_dependent_columns) - len([independent_column])

        dependent_columns = [i for i in np.random.choice(np.delete(np.arange(self.dataset.shape[1]), np.append(independent_column, base_dependent_columns)), size=size, replace=False)]
        for add_column in dependent_columns:
            extended_x = self.dataset[:, base_dependent_columns.tolist()+[add_column]]
            score = self.regression_score(extended_x, base_y)

            if base_r2_score < 0 or score < 0:
                raise Exception('Some score is <0. Think about how to calc score difference!')

            if score>base_r2_score:
                _score =  score-base_r2_score #adding a_column helps
            else:
                _score = -(score-base_r2_score) #adding a_column does not help

            add_columns.append([self.dataset[:, add_column], _score])




        return batchify(base_x,  add_columns, base_y, config["budget_join"])
