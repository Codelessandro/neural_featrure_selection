import numpy as np
from config import *
from scipy.stats import pearsonr

from utils import *
from timeit import default_timer as timer
from matplotlib import pyplot as plt


def pearson(base_dataset,add_column,y):
    return pearsonr(add_column,y)

def constant(base_dataset,add_column,y):
    return 0.5


def evaluation(base_dataset, add_column, y, neural_feature_model):
    evaluations=[]
    if config["machine_learning_task"]=="regression":
        evaluations.append( {
            "constant" : constant(base_dataset,add_column, y)
        })
        p_start = timer()
        p = pearson(base_dataset, add_column, y)[0]
        p_end = timer()
        evaluations.append( {
            "pearson": p
        })


    extended_x =  np.concatenate((base_dataset,add_column.reshape(-1,1)),axis=1)  #.reshape(1,config["max_limit_dataset_rows"]*(config["nr_base_columns"]+1))
    x_batch = generate_batch(extended_x)

    nfs_start = timer()
    scores=neural_feature_model.predict(x_batch)
    score = np.mean(scores)
    nfs_end = timer()

    print("For this dataset with this column we have the following scores:")

    for e in evaluations:
        method = list(e.keys())[0]
        print("method:" + str(method) + str(e[method]))

    print("Neural Feature Selection:" + str(score))

    return score,  (nfs_end - nfs_start), pearson(base_dataset, add_column, y)[0], (p_end-p_start)




def plot_performance(nfs, method):

    N=np.arange(len(method))

    width = 1
    p1 = plt.bar(N*2, nfs, width, align='edge')
    print(nfs)
    print(method)
    p2 = plt.bar(N*2+1, method, width, align='edge')

    plt.xlabel('additional Column on a Dataset')
    plt.ylabel('Time in seconds')
    plt.yscale('log')

    plt.legend((p1[0], p2[0]), ('NFS', 'Pearson'))
    plt.show()
