from DummyData import *
from DataSetWine import *
from BaseData import *
from config import *

import math
import pdb

from evaluation import *
from model_feedforward import *
from functools import reduce



np.set_printoptions(suppress=True)


def merge_data_sets(datasets):
    for d in datasets:
        d.load()


    y_score = []
    xy      = []

    for d in datasets:
        y_score.append(d.y_score)
        xy.append(d.xy)

    y_score = np.concatenate(y_score)
    xy = np.concatenate(xy)


    return xy, y_score


WineData = BaseData('data/winequality-red.csv', ';', 11, 10, config["nr_base_columns"], rifs=True)

GoogleData = BaseData('data/google-safe-browsing-transparency-report-data.csv', ',', 10, 10, config["nr_base_columns"], rifs=True)
CampusData = BaseData('data/placement_data_full_class.csv', ',', 14, 10, config["nr_base_columns"], rifs=True)
FootballData = BaseData('data/results_football.csv', ',', 3, 10, config["nr_base_columns"], rifs=True)
KingSalesData = BaseData('data/kc_house_data.csv', ',', 2, 10, config["nr_base_columns"], rifs=True)
AvocadoSalesData = BaseData('data/avocado.csv', ',', 2, 10, config["nr_base_columns"], rifs=True)
TeslaStocksData = BaseData('data/TSLA.csv', ',', 6, 10, config["nr_base_columns"], rifs=True)
WeatherHistoryData = BaseData('data/weatherHistory.csv', ',', 8, 10, config["nr_base_columns"], rifs=True)
VoiceData = BaseData('data/voice.csv', ',', 19, 10, config["nr_base_columns"], rifs=True)

xy, y_score = merge_data_sets([WineData, GoogleData, CampusData, FootballData, KingSalesData, AvocadoSalesData, TeslaStocksData, WeatherHistoryData, VoiceData])

'''
from scipy.stats import pearsonr
pearsons=[]
nfs=[]

for  i in np.arange(y_score.shape[0]):
    pearson = pearsonr(x[i][:,5], y_data[i])[0]
    _nfs = y_score[i]
    pearsons.append(pearson)
    nfs.append(_nfs)
'''

model, i, modelhistory = best_feedforward_model(xy, y_score, True)





if config["budget_join"]:
    add_eval_columns = [
        WineData.base_dataset[:, 0],
        WineData.base_dataset[:, 1],
        WineData.base_dataset[:, 2],
        WineData.base_dataset[:, 3],
        WineData.base_dataset[:, 11],
    ]

    for i in np.arange(135):
        add_eval_columns=add_eval_columns.append(WineData.base_dataset[:, 0])


    evaluations=evaluation(
        WineData.base_xy,
        add_eval_columns,
        WineData.base_dataset[:, 11],
        model
    )
    plot_performance([evaluations[1]], [evaluations[3]])

else:
    evaluations = []

    print("i:")
    print(i)

    print("++++")
    print("++++")
    evaluations.append(evaluation(WineData.base_xy, [np.random.normal(0, 10, WineData.base_dataset[:, 9].shape[0])],
                                  WineData.base_dataset[:, 11], model))
    print("++++")
    print("++++")
    for c in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        evaluations.append(
            evaluation(WineData.base_xy, [WineData.base_dataset[:, c]], WineData.base_dataset[:, 11], model))
    print("++++")
    print("++++")
    evaluations.append(evaluation(WineData.base_xy, [WineData.base_dataset[:, 11]], WineData.base_dataset[:, 11], model))
    print("++++")
    print("++++")

    evaluations = np.array(evaluations)
    print(evaluations)

    import pdb; pdb.set_trace()
    plt.scatter(np.concatenate((evaluations[:, 0])), np.concatenate((evaluations[:, 2])))
    plt.xlabel('NFS Score')
    plt.ylabel('Pearson')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()
    import pdb; pdb.set_trace()
    plot_performance(evaluations[:, 1], evaluations[:, 3])

