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


def evaluation(base_dataset, add_columns, y, neural_feature_model):

    evaluations=[]
    if config["machine_learning_task"]=="regression":

        '''
        evaluations.append( {
            "constant" : constant(base_dataset,add_column, y)
        })
        '''


        p_start = timer()
        for c in add_columns:
            p = pearson(base_dataset, c, y)[0]
            evaluations.append( {
                "pearson": p
            })
        p_end = timer()


    extended_x =  np.concatenate((base_dataset,  np.vstack(add_columns).reshape(-1,len(add_columns))  ), axis=1) #.reshape(1,config["max_limit_dataset_rows"]*(config["nr_base_columns"]+1))
    x_batch = generate_batch(extended_x)

    nfs_start = timer()
    scores=neural_feature_model.predict(x_batch)
    scores = np.mean(scores,axis=0)
    nfs_end = timer()

    '''
    print("For this dataset with this column we have the following scores:")

    for e in evaluations:
        method = list(e.keys())[0]
        print("method:" + str(method) + str(e[method]))

    print("Neural Feature Selection:" + str(score))
    '''

    evaluations = list(map(lambda e: e['pearson'], evaluations))
    return  scores,  (nfs_end - nfs_start), evaluations, (p_end-p_start)




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
