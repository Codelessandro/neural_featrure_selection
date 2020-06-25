from enum import Enum


class Task(Enum):
    regression = 1
    classification = 2
    multivariate_time_series = 3


config = {
    "machine_learning_task": "regression",  # classification
    "nr_dummy_datasets": 5,
    "max_limit_dataset_rows": 1000,
    "dataset_rows": 1000,
    "batch_size": 20,  # nr_# rows
    "nr_base_columns": 5,
    "nr_feedforward_iterations": 1,
    "budget_join": True,
    "nr_add_columns_budget": 140,
    # if this number is biger than colums we can get from the dataset, bootstrapping is activated automatically #can also be set to 0
    "task": Task.regression,
    "prod" : False, #production True means that full dataset will be loaded
    "nfs_output_threshold": 0.5, #between 0 and 1 - nfs scores that are above this value will be taken to augment datasert
    "multivariate_time_series": {
        "window_size": 10,
        "offset": 0,  # 0 is the next after the x batch
        "test_split": 0.2
    },

}
