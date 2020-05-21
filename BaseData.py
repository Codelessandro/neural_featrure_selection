import pdb
import pandas as pd
from numpy import genfromtxt
from sklearn import datasets, linear_model
import numpy as np
#import pandas as pd
from utils import *
from numpy.random import default_rng

from config import *

class BaseData():
    def __init__(self, dataset, delimiter, target, random, base_size=5, rifs=False, datetime=0):
        self.dataset = dataset
        self.delimiter = delimiter
        self.target = target
        self.random = random
        self.base_size = base_size
        self.rifs = rifs
        self.datetime = datetime

    def regression_score(self, x,y):
        regr = linear_model.LinearRegression()
        regr.fit(x,y)
        return regr.score(x,y)

    def load(self):
        dataset = genfromtxt(self.dataset, delimiter=self.delimiter)[1:]
        dataset = dataset[0:config["max_limit_dataset_rows"]]
        self.dataset = np.nan_to_num(dataset)
        if self.datetime > 0:
            dataset[:, self.datetime] = np.datetime64(dataset[:, self.datetime]).astype(object).day
        self.perm = self.permutation(self.dataset, self.target, self.random)

    def permutation(self, dataset, target, random):
        rng = default_rng()
        for _ in range(random):
            random_col =  np.random.choice(np.delete(np.arange(dataset.shape[1]),target),size=self.base_size)
            self.data = self.generate_data(random_col, self.target, self.random)

    def generate_data(self, base_dependent_columns, independent_column, random):
        rng = default_rng()

        base_x = self.dataset[:, base_dependent_columns]
        self.base_x = base_x
        base_y = self.dataset[:, independent_column]
        self.base_y = base_y

        if self.rifs is True:
            random_column = rng.choice(int(np.amax(self.dataset)), size=self.dataset.shape[0], replace=True)
            rifs_base_x = np.concatenate((base_x, random_column[:, None]), axis=1)
            base_r2_score = self.regression_score(rifs_base_x, base_y)
        else:
            base_r2_score = self.regression_score(base_x, base_y)

        add_columns = []
        rng = default_rng()

        #dependent_columns = [6, 7, 8, 9, 10]
        dependent_columns = [i for i in rng.choice(self.dataset.shape[1], size=self.dataset.shape[1]-self.base_size, replace=False) if i not in base_dependent_columns+[independent_column]]

        for add_column in dependent_columns:
            extended_x = self.dataset[:, base_dependent_columns.tolist()+[add_column]]
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

#GoogleData = BaseData('data/google-safe-browsing-transparency-report-data.csv', ',', 10, 1, rifs=True)
#GoogleData.load()
