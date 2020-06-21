from enum import Enum
class Task(Enum):
    regression = 1
    classification = 2
    multivariate_time_series = 3


config = {
    "machine_learning_task" : "regression", #classification
    "nr_dummy_datasets" : 5,
    "max_limit_dataset_rows" : 1000,
    "dataset_rows" : 1000,
    "batch_size": 20, #nr_# rows
    "nr_base_columns" : 5,
    "nr_feedforward_iterations": 10,
    "budget_join" : False,
    "nr_add_columns_budget": 5, #if this number is biger than colums we can get from the dataset, bootstrapping is activated automatically
    "task": Task.regression
}
