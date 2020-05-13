import pdb
import pandas as pd
from numpy import genfromtxt
from sklearn import datasets, linear_model
import numpy as np
#import pandas as pd
from utils import *
from numpy.random import default_rng

class BaseData():
    def __init__(self, dataset, delimiter, target, random, datetime=0):
        self.dataset = dataset
        self.delimiter = delimiter
        self.target = target
        self.random = random
        self.datetime = datetime

    def regression_score(self, x,y):
        regr = linear_model.LinearRegression()
        regr.fit(x,y)
        return regr.score(x,y)

    def load(self):
        dataset = genfromtxt(self.dataset, delimiter=self.delimiter)[1:]
        self.dataset = np.nan_to_num(dataset)
        if self.datetime > 0:
            dataset[:, self.datetime] = np.datetime64(dataset[:, self.datetime]).astype(object).day
        self.perm = self.permutation(self.dataset, self.target, self.random)

    def permutation(self, dataset, target, random):
        rng = default_rng()
        for _ in range(random):
            random_col = [i for i in rng.choice(dataset.shape[1]-1, size=5, replace=False)]
            self.data = self.generate_data(random_col, self.target, self.random)

    def generate_data(self, base_dependent_columns, independent_column, random):
        base_x = self.dataset[:, base_dependent_columns]
        base_y = self.dataset[:, independent_column]
        base_r2_score = self.regression_score(base_x,base_y)

        add_columns = []
        #dependent_columns = np.setdiff1d(self.dataset, self.dataset[:, base_dependent_columns])
        rng = default_rng()
        size = self.dataset.shape[1]-5
        for _ in range(random):
            dependent_columns = [i for i in rng.choice(self.dataset.shape[1]-1, size=size, replace=False)]
        #dependent_columns = [6, 7, 8, 9, 10]
        for add_column in dependent_columns:
            extended_x = self.dataset[:, base_dependent_columns+[add_column]]
            score = self.regression_score(extended_x, base_y)

            if base_r2_score < 0 or score < 0:
                raise Exception('Some score is <0. Think about how to calc score difference!')

            if score>base_r2_score:
                _score =  score-base_r2_score #adding a_column helps
            else:
                _score = -(score-base_r2_score) #adding a_column does not help


            print(_score)
            add_columns.append([self.dataset[:,add_column],_score])

        return to_row([[base_x, add_columns]])

#WineData = BaseData('data/winequality-red.csv', ';', 11, 1)
#WineData.load()

#GoogleData = BaseData('data/google-safe-browsing-transparency-report-data.csv', ',', 10, 1)
#GoogleData.load()