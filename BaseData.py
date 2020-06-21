import pdb
import pandas as pd
from numpy import genfromtxt
from sklearn import datasets, linear_model
import numpy as np
from utils import *
from numpy.random import default_rng
from sklearn.utils import resample


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
            self.x = np.empty(shape=[1, config['batch_size'], config['nr_base_columns']+ config['nr_add_columns_budget'] ])
            self.y_score = np.empty(shape=[1,config['nr_add_columns_budget'],])
        else:
            self.x = np.empty(shape=[1, config['batch_size'], config['nr_base_columns']+1])
            self.y_score = np.empty(shape=[1])


        boostrapped_columns_needed = config["nr_add_columns_budget"] - (self.dataset.shape[1] - self.base_size -1)

        if boostrapped_columns_needed>0:
            dataset_flattened = self.dataset.flatten()

        for _ in range(boostrapped_columns_needed):
            boot = resample(dataset_flattened, replace=True, n_samples=self.dataset.shape[0], random_state=1)
            boot_ext = np.expand_dims(boot, axis=1)
            self.dataset = np.append(self.dataset, boot_ext, axis=1)


        for _ in range(combine):
            random_col = np.random.choice(np.delete(np.arange(dataset.shape[1]),target),size=self.base_size,replace=False)
            x, y_data, y_score = self.generate_data(random_col, self.target)

            self.x = np.concatenate((self.x, x))

            try:
                self.y_data = np.concatenate((self.y_data, y_data))
                self.y_score = np.concatenate((self.y_score, y_score))
            except:
                pdb.set_trace()


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
        #pdb.set_trace()
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

#google = BaseData('data/google-safe-browsing-transparency-report-data.csv', ',', 10, 10, config["nr_base_columns"], rifs=True)
#google.load()
#print(google.xy.shape)

#campus_placement = BaseData('data/placement_data_full_class.csv', ',', 14, 10, config["nr_base_columns"], rifs=True)
#campus_placement.load()
#print(campus_placement.xy)

#added here decoding parameter to getfromtxt
#football_results = BaseData('data/results_football.csv', ',', 3, 10, config["nr_base_columns"], rifs=True)
#football_results.load()
#print(football_results.xy)

#new method to replace blank cells has to be added
#games_sales = BaseData('data/Video_Games_Sales_as_at_22_Dec_2016.csv', ',', 15, 10, config["nr_base_columns"], rifs=True)
#games_sales.load()
#print(games_sales.xy)

#king_sales = BaseData('data/kc_house_data.csv', ',', 2, 10, config["nr_base_columns"], rifs=True)
#king_sales.load()
#print(king_sales.xy)

#avocado_sales = BaseData('data/avocado.csv', ',', 2, 10, config["nr_base_columns"], rifs=True)
#avocado_sales.load()
#print(avocado_sales.xy)

#too many text columns
#brazil_rent = BaseData('data/houses_to_rent_brazil.csv', ',', 12, 10, config["nr_base_columns"], rifs=True)
#brazil_rent.load()
#print(brazil_rent.xy)

#tesla_stocks = BaseData('data/TSLA.csv', ',', 6, 10, config["nr_base_columns"], rifs=True)
#tesla_stocks.load()
#print(tesla_stocks.xy)

#weatherHistory = BaseData('data/weatherHistory.csv', ',', 8, 10, config["nr_base_columns"], rifs=True)
#weatherHistory.load()
#print(weatherHistory.xy)

#voice = BaseData('data/voice.csv', ',', 19, 10, config["nr_base_columns"], rifs=True)
#voice.load()
#print(voice.xy)

#countries_of_the_world = BaseData('data/countries_of_the_world.csv', ',', 14, 10, config["nr_base_columns"], rifs=True)
#countries_of_the_world.load()
#print(countries_of_the_world.xy)
