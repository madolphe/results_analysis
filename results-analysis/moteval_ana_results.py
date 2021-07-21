import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import *


def compute_mean_per_condition(row):
    """
    3 conditions for MOT: speed=1,4 or 8
    Compute accuracy for each condition
    """
    dict_mean_accuracy_per_condition = {}
    dict_mean_rt_per_condition = {}
    for idx, condition_key in enumerate(row['results_speed_stim']):
        if condition_key not in dict_mean_accuracy_per_condition:
            dict_mean_accuracy_per_condition[condition_key] = []
            dict_mean_rt_per_condition[condition_key] = []
        dict_mean_accuracy_per_condition[condition_key].append(float(row['results_correct'][idx]))
        dict_mean_rt_per_condition[condition_key].append(float(row['results_rt'][idx]))
    for key in dict_mean_accuracy_per_condition.keys():
        row[f"{key}-RT"] = np.mean(dict_mean_rt_per_condition[key])
        row[f"{key}-accuracy"] = np.mean(dict_mean_accuracy_per_condition[key])
    return row


if __name__ == '__main__':
    csv_path = "../outputs/moteval/moteval.csv"
    dataframe = pd.read_csv(csv_path)
    dataframe = dataframe.apply(lambda row: transform_str_to_list(row, [
        'results_responses', 'results_rt', 'results_speed_stim', 'results_correct']), axis=1)
    dataframe = delete_uncomplete_participants(dataframe)
    dataframe = dataframe.apply(compute_mean_per_condition, axis=1)
    dataframe.to_csv('../outputs/moteval/moteval_treat.csv')
