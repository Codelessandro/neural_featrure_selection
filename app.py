from DummyData import *
from DataSetWine import *
from BaseData import *
from config import *

import math
import pdb

from evaluation import *
from model_feedforward import *
from functools import reduce
from load_data import *

print("a")
np.set_printoptions(suppress=True)
print("b")


xy, y_score = load_data(config["task"])
print("c1")

#y_score, best_score = normalize(y_score)
print("c2")

model, i, modelhistory = best_feedforward_model(xy, y_score, True)
print("d")



print("We take:")
print(i)

print("Best score for normalization scale:")
#print(best_score)

WineData = BaseData('data/winequality-red.csv', ';', 11, 10, config["nr_base_columns"], rifs=True)
WineData.load()
_=evaluation_wrapper(config["task"], model, 'data/winequality-red.csv', True, WineData, columns=[0, 1, 2, 3, 11], target=11)


GoogleData = BaseData('data/google-safe-browsing-transparency-report-data.csv', ',', 10, 10, config["nr_base_columns"],
                      rifs=True)
GoogleData.load()
_=evaluation_wrapper(config["task"], model, 'data/google-safe-browsing-transparency-report-data.csv', True, GoogleData, columns=[0, 1, 2, 3, 10], target=10)


CampusData = BaseData('data/placement_data_full_class.csv', ',', 14, 10, config["nr_base_columns"], rifs=True,
                      text_columns=[1, 3, 5, 6, 8, 9, 11, 13])
CampusData.load()
_=evaluation_wrapper(config["task"], model, 'data/placement_data_full_class.csv', True, CampusData, columns=[0, 1, 2, 3, 14], target=14)

FootballData = BaseData('data/results_football.csv', ',', 3, 10, config["nr_base_columns"], rifs=True,
                        text_columns=[1, 2, 5, 6, 7, 8], date=0)
FootballData.load()
_=evaluation_wrapper(config["task"], model, 'data/results_football.csv', True, FootballData, columns=[0, 1, 2, 3, 8], target=8)

KingSalesData = BaseData('data/kc_house_data.csv', ',', 2, 10, config["nr_base_columns"], rifs=True, date=1)
KingSalesData.load()
_=evaluation_wrapper(config["task"], model, 'data/kc_house_data.csv', True, KingSalesData, columns=[0, 1, 2, 3, 20], target=20)

AvocadoSalesData = BaseData('data/avocado.csv', ',', 2, 10, config["nr_base_columns"], rifs=True, text_columns=[11, 13],
                            date=1)
AvocadoSalesData.load()
_=evaluation_wrapper(config["task"], model, 'data/avocado.csv', True, AvocadoSalesData, columns=[0, 1, 2, 3, 13], target=13)

BrazilRentData = BaseData('data/houses_to_rent.csv', ',', 12, 10, config["nr_base_columns"], rifs=True,
                          text_columns=[6, 7])
BrazilRentData.load()
_=evaluation_wrapper(config["task"], model, 'data/houses_to_rent.csv', True, BrazilRentData, columns=[0, 1, 2, 3, 12], target=12)

TeslaStocksData = BaseData('data/TSLA.csv', ',', 6, 10, config["nr_base_columns"], rifs=True, date=0)
TeslaStocksData.load()
_=evaluation_wrapper(config["task"], model, 'data/TSLA.csv', True, TeslaStocksData, columns=[0, 1, 2, 3, 6], target=6)

WeatherHistoryData = BaseData('data/weatherHistory.csv', ',', 8, 10, config["nr_base_columns"], rifs=True,
                              text_columns=[1, 2, 11], date=0)
WeatherHistoryData.load()
_=evaluation_wrapper(config["task"], model, 'data/weatherHistory.csv', True, WeatherHistoryData, columns=[0, 1, 2, 3, 11], target=11)

VoiceData = BaseData('data/voice.csv', ',', 19, 10, config["nr_base_columns"], rifs=True, text_columns=[20])
VoiceData.load()
_=evaluation_wrapper(config["task"], model, 'data/voice.csv', True, VoiceData, columns=[0, 1, 2, 3, 19], target=19)

CountriesData = BaseData('data/countries_of_the_world.csv', ',', 14, 10, config["nr_base_columns"], rifs=True,
                         text_columns=[0, 1])
CountriesData.load()
_=evaluation_wrapper(config["task"], model, 'data/countries_of_the_world.csv', True, CountriesData, columns=[0, 1, 2, 3, 19], target=19)

HeartData = BaseData('data/heart.csv', ',', 13, 10, config["nr_base_columns"], rifs=True)
HeartData.load()
_=evaluation_wrapper(config["task"], model, 'data/heart.csv', True, HeartData, columns=[0, 1, 2, 3, 13], target=13)

StudentsData = BaseData('data/StudentsPerformance.csv', ',', 7, 10, config["nr_base_columns"], rifs=True,
                        text_columns=[0, 1, 2, 3, 4])
StudentsData.load()
_=evaluation_wrapper(config["task"], model, 'data/StudentsPerformance.csv', True, StudentsData, columns=[0, 1, 2, 3, 7], target=7)

HRData = BaseData('data/WA_Fn-UseC_-HR-Employee-Attrition.csv', ',', 3, 10, config["nr_base_columns"], rifs=True,
                  text_columns=[1, 2, 4, 7, 11, 15, 17, 21, 22])
HRData.load()
_=evaluation_wrapper(config["task"], model, 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv', True, HRData, columns=[0, 1, 2, 3, 34], target=34)

InsuranceData = BaseData('data/insurance.csv', ',', 6, 10, config["nr_base_columns"], rifs=True, text_columns=[1, 4, 5])
InsuranceData.load()
_=evaluation_wrapper(config["task"], model, 'data/insurance.csv', True, InsuranceData, columns=[0, 1, 2, 3, 6], target=6)

FreedomData = BaseData('data/hfi_cc_2019.csv', ',', 4, 10, config["nr_base_columns"], rifs=True, text_columns=[1, 2, 3])
FreedomData.load()
_=evaluation_wrapper(config["task"], model, 'data/hfi_cc_2019.csv', True, FreedomData, columns=[0, 1, 2, 3, 119], target=119)

GlassData = BaseData('data/glass.csv', ',', 9, 10, config["nr_base_columns"], rifs=True)
GlassData.load()
_=evaluation_wrapper(config["task"], model, 'data/glass.csv', True, GlassData, columns=[0, 1, 2, 3, 9], target=9)

AcquisitionData = BaseData('data/acquisitions.csv', ',', 7, 10, config["nr_base_columns"], rifs=True,
                           text_columns=[0, 1, 4, 5, 6, 8, 9])
AcquisitionData.load()
_=evaluation_wrapper(config["task"], model, 'data/acquisitions.csv', True, AcquisitionData, columns=[0, 1, 2, 3, 9], target=9)

SecomData = BaseData('data/uci-secom.csv', ',', 591, 10, config["nr_base_columns"], rifs=True, date=0)
SecomData.load()
_=evaluation_wrapper(config["task"], model, 'uci-secom.csv', True, SecomData, columns=[0, 1, 2, 3, 591], target=591)
