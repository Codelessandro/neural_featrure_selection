import pdb
import pandas as pd
from numpy import genfromtxt
from sklearn import datasets, linear_model
import numpy as np
#import pandas as pd
from utils import *
from numpy.random import default_rng

np.random.seed(0)

from config import *

class BaseData():
    def __init__(self, dataset_path, delimiter, target, random, base_size=5, datetime=0):

        self.augmented_nr_rows = config["dataset_rows"]

        self.dataset_path = dataset_path
        self.delimiter = delimiter
        self.target = target
        self.random = random
        self.base_size = base_size
        self.datetime = datetime


    def regression_score(self, x,y):
        regr = linear_model.LinearRegression()
        regr.fit(x,y)
        return regr.score(x,y)

    def fix_row_size(self):
        if self.dataset.shape[0]>config["dataset_rows"]:
            self.dataset = self.dataset[0:config["dataset_rows"]]
        else:
            nr_repeats = round(config["dataset_rows"] / self.dataset.shape[0]+0.5)
            self.dataset = np.repeat(self.dataset, nr_repeats, axis=0)
            self.dataset = self.dataset[0:config["dataset_rows"]]



    def load(self):
        self.dataset = genfromtxt(self.dataset_path, delimiter=self.delimiter)[1:]

        self.dataset = np.nan_to_num(self.dataset)
        if self.datetime > 0:
            self.dataset[:, self.datetime] = np.datetime64(dataset[:, self.datetime]).astype(object).day
        #self.perm = self.permutation(self.dataset, self.target, self.random)

        random_col = np.random.choice(np.delete(np.arange(self.dataset.shape[1]), self.target), size=self.base_size, replace=False)
        self.base_dataset = self.dataset
        self.x, self.y_data, self.y_score = self.generate_data(random_col, self.target)
        self.xy = np.concatenate( (self.x,np.expand_dims(self.y_data,axis=2)),axis=2 )



    def permutation(self, dataset, target, random):
        for _ in range(random):
            random_col =  np.random.choice(np.delete(np.arange(dataset.shape[1]),target),size=self.base_size,replace=False)
            self.data = self.generate_data(random_col, self.target)

    def generate_data(self, base_dependent_columns, independent_column):
        base_x = self.dataset[:, base_dependent_columns]
        self.base_x = base_x
        base_y = self.dataset[:, independent_column]
        self.base_y = base_y
        self.base_xy = np.concatenate((self.base_x, np.expand_dims(self.base_y, axis=1)), axis=1)
        base_r2_score = self.regression_score(base_x,base_y)

        add_columns = []
        rng = default_rng()
        #for _ in range(random):
            #dependent_columns = [i for i in rng.choice(self.dataset.shape[1]-1, size=size, replace=False)]
        #dependent_columns = [6, 7, 8, 9, 10]
        #dependent_columns =np.delete(np.delete(np.arange(self.dataset.shape[1]), independent_column), base_dependent_columns)
        dependent_columns =np.delete(np.arange(self.dataset.shape[1]), base_dependent_columns)


        for add_column in dependent_columns:
            extended_x = self.dataset[:, base_dependent_columns.tolist()+[add_column]]
            score = self.regression_score(extended_x, base_y)

            if base_r2_score < 0 or score < 0:
                raise Exception('Some score is <0. Think about how to calc score difference!')

            if score>base_r2_score:
                _score =  score-base_r2_score #adding a_column helps
            else:
                _score = -(score-base_r2_score) #adding a_column does not help

            add_columns.append([self.dataset[:,add_column],_score])

        return batchify(base_x,  add_columns, base_y)

