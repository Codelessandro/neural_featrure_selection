import numpy as np
from BaseData import *
from config import *


def merge_data_sets(datasets):
    for d in datasets:
        d.load()

    y_score = []
    xy = []

    for d in datasets:
        y_score.append(d.y_score)
        xy.append(d.xy)

    y_score = np.concatenate(y_score)
    xy = np.concatenate(xy)

    return xy, y_score


def load_data(task):
    if task == Task.regression:
        WineData = BaseData('data/winequality-red.csv', ';', 11, 10, config["nr_base_columns"], rifs=True)
        GoogleData = BaseData('data/google-safe-browsing-transparency-report-data.csv', ',', 10, 10,
                              config["nr_base_columns"], rifs=True)
        CampusData = BaseData('data/placement_data_full_class.csv', ',', 14, 10, config["nr_base_columns"], rifs=True)
        FootballData = BaseData('data/results_football.csv', ',', 3, 10, config["nr_base_columns"], rifs=True)
        KingSalesData = BaseData('data/kc_house_data.csv', ',', 2, 10, config["nr_base_columns"], rifs=True)
        AvocadoSalesData = BaseData('data/avocado.csv', ',', 2, 10, config["nr_base_columns"], rifs=True)
        TeslaStocksData = BaseData('data/TSLA.csv', ',', 6, 10, config["nr_base_columns"], rifs=True)
        WeatherHistoryData = BaseData('data/weatherHistory.csv', ',', 8, 10, config["nr_base_columns"], rifs=True)
        VoiceData = BaseData('data/voice.csv', ',', 19, 10, config["nr_base_columns"], rifs=True)

        if config["prod"]==True:
            xy, y_score = merge_data_sets(
                [WineData, GoogleData, CampusData, FootballData, KingSalesData, AvocadoSalesData, TeslaStocksData,
                 WeatherHistoryData, VoiceData])

        if config["prod"]==False:
            xy, y_score = merge_data_sets([WineData, GoogleData])

    if task == Task.multivariate_time_series:
        BirthDeaths3 = BaseData('data/multivariate_time_series/_births_and_deaths.csv', ';', 3, 1, base_size=config["nr_base_columns"])
        BirthDeaths4 = BaseData('data/multivariate_time_series/_births_and_deaths.csv', ';', 4, 1, base_size=config["nr_base_columns"])

        xy, y_score = merge_data_sets([BirthDeaths3, BirthDeaths4])


    if task == Task.classification:
        pass

    return xy, y_score
