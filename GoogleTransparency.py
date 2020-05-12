import pdb
import pandas as pd
from numpy import genfromtxt
from sklearn import datasets, linear_model
import numpy as np
from utils import *


class GoogleTransparency():
    def __init__(self):
        self.load()

    def regression_score(self, x,y):
        regr = linear_model.LinearRegression()
        regr.fit(x,y)
        return regr.score(x,y)


    def load(self):
        dataset = genfromtxt('data/google-safe-browsing-transparency-report-data_nan.csv', delimiter=',')[1:]
        self.dataset=dataset
        self.data = self.generate_data( [0,1,2,3,4,5], 12)

    def generate_data(self, base_dependent_columns, independent_column):
        base_x = self.dataset[:,base_dependent_columns]
        base_y = self.dataset[:,independent_column]
        base_r2_score = self.regression_score(base_x,base_y)

        add_dependent_columns = [6,7,8,9,10,11]
        add_columns = []
        for add_column in add_dependent_columns:
            extended_x = self.dataset[:,base_dependent_columns+[add_column]]
            score = self.regression_score(extended_x, base_y)

            if base_r2_score < 0 or score < 0:
                raise Exception('Some score is <0. Think about how to calc score difference!')

            if score>base_r2_score:
                _score =  score-base_r2_score #add_column helps
            else:
                _score = -(score-base_r2_score) #add_column is worse


            print(_score)
            add_columns.append([self.dataset[:,add_column],_score])

        return to_row([[base_x, add_columns]])
