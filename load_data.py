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
        WineData = BaseData('data/winequality-red.csv', ';', 7, 10, config["nr_base_columns"], rifs=True)
        GoogleData = BaseData('data/google-safe-browsing-transparency-report-data.csv', ',', 10, 10, config["nr_base_columns"], rifs=True)
        CampusData = BaseData('data/placement_data_full_class.csv', ',', 14, 10, config["nr_base_columns"], rifs=True, text_columns=[1, 3, 5, 6, 8, 9, 11, 13])
        FootballData = BaseData('data/results_football.csv', ',', 3, 10, config["nr_base_columns"], rifs=True, text_columns=[1, 2, 5, 6, 7, 8], date=0)
        KingSalesData = BaseData('data/kc_house_data.csv', ',', 2, 10, config["nr_base_columns"], rifs=True, date=1)
        AvocadoSalesData = BaseData('data/avocado.csv', ',', 2, 10, config["nr_base_columns"], rifs=True, text_columns=[11, 13], date=1)
        Brazil_Rent = BaseData('data/houses_to_rent.csv', ',', 12, 10, config["nr_base_columns"], rifs=True, text_columns=[6, 7])
        TeslaStocksData = BaseData('data/TSLA.csv', ',', 6, 10, config["nr_base_columns"], rifs=True, date=0)
        WeatherHistoryData = BaseData('data/weatherHistory.csv', ',', 8, 10, config["nr_base_columns"], rifs=True, text_columns=[1,2, 11], date=0)
        VoiceData = BaseData('data/voice.csv', ',', 19, 10, config["nr_base_columns"], rifs=True, text_columns=[20])
        CountriesData = BaseData('data/countries_of_the_world.csv', ',', 14, 10, config["nr_base_columns"], rifs=True, text_columns=[0,1])

        train_datasets = [WineData, GoogleData, CampusData, FootballData, KingSalesData, AvocadoSalesData, Brazil_Rent,
                          TeslaStocksData, WeatherHistoryData, VoiceData, CountriesData]

        if config["prod"]==False:
            train_datasets = train_datasets[0:2]

        xy, y_score = merge_data_sets(train_datasets)
        config["current_dataset_names"] = list(map(lambda d: d.dataset_path, train_datasets))



    if task == Task.multivariate_time_series:
        BirthDeaths3 = BaseData('data/multivariate_time_series/_births_and_deaths.csv', ';', 3, 1, base_size=config["nr_base_columns"])
        BirthDeaths4 = BaseData('data/multivariate_time_series/_births_and_deaths.csv', ';', 4, 1, base_size=config["nr_base_columns"])

        xy, y_score = merge_data_sets([BirthDeaths3, BirthDeaths4])


    if task == Task.classification:
        pass


    return xy, y_score
