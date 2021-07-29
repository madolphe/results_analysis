import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cal_stan_accuracy_rt import CalStan_accuracy, CalStan_rt
from utils import *


def compute_nearfarcond(row, ind_nearfar):
    """
        From the row of results, return the list of farcondition if elt is min/max in results_targetvalue
        The ind_nearfar 0 means near and 1 means far conditions.
    """
    results_responses = list(row["results_responses_pos"])
    results_targetvalue = list(row["results_target_distance"])
    min_tmp = min(results_targetvalue)
    targind_tmp = []
    targind_tmp = [0 if t == min_tmp else 1 for t in results_targetvalue]
    out = [results_responses[idx] for idx, elt in enumerate(targind_tmp) if elt == ind_nearfar]

    return np.array(out)


def parse_to_int(elt: str) -> int:
    """
        Parse string value into int, if string is null parse to 0
        If null; participant has not pressed the key when expected
    """
    if elt == '':
        return 0
    return int(elt)


def transform_string_to_row(row, column):
    return [int(elt) for elt in row[column].split(',') if elt]


def compute_sum_to_row(row, column):
    return np.sum(row[column])


if __name__ == '__main__':
    csv_path = "../outputs/loadblindness/loadblindness.csv"
    dataframe = pd.read_csv(csv_path)
    dataframe = delete_uncomplete_participants(dataframe)
    dataframe["results_responses_pos"] = dataframe.apply(
        lambda row: transform_string_to_row(row, "results_responses_pos"),
        axis=1)
    dataframe["results_target_distance"] = dataframe.apply(
        lambda row: transform_string_to_row(row, "results_target_distance"),
        axis=1)
    # extract far
    dataframe['far_response'] = dataframe.apply(lambda row: compute_nearfarcond(row, 1), axis=1)
    dataframe['near_response'] = dataframe.apply(lambda row: compute_nearfarcond(row, 0), axis=1)
    dataframe['sum_far'] = dataframe.apply(lambda row: compute_sum_to_row(row, "far_response"), axis=1)
    dataframe['sum_near'] = dataframe.apply(lambda row: compute_sum_to_row(row, "near_response"), axis=1)
    dataframe['total_resp'] = dataframe.apply(lambda row: 20, axis=1)
    dataframe['accuracy_near'] = dataframe['sum_near'] / dataframe['near_response'].apply(lambda row: len(row))
    dataframe['accuracy_far'] = dataframe['sum_far'] / dataframe['far_response'].apply(lambda row: len(row))

    # -------------------------------------------------------------------#
    # Latent factor analysis
    # sumirize two days experiments
    sum_observers = []
    conditions_names = ['accuracy_near', 'accuracy_far']
    # extract observer index information
    indices_id = extract_id(dataframe, num_count=2)
    for ob in indices_id:
        print(ob)
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        sum_observers.append([np.mean(tmp_df.accuracy_near), np.sum(tmp_df.accuracy_far)])
    sum_observers = pd.DataFrame(sum_observers, conditions_names)
    # for save summary data
    sum_observers['total_resp'] = sum_observers.apply(lambda row: 40, axis=1)  # two days task
    sum_observers.to_csv('../outputs/loadblindness/sumdata_loadblindness.csv', header=True, index=False)
    # -------------------------------------------------------------------#

    # -------------------------------------------------------------------#
    # BAYES ANALYSIS
    nb_trials = len(dataframe['near_response'][0])
    stan_distributions = get_stan_accuracy_distributions(dataframe, conditions_names, nb_trials)
    # Draw figures for accuracy data
    plot_args = {'list_xlim': [0.5, 2.5], 'list_ylim': [0, 1],
                 'list_set_xticklabels': ['Near', 'Far'], 'list_set_xticks': [1, 2],
                 'list_set_yticklabels': ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                 'list_set_yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0], }
    plot_all_accuracy_figures(stan_distributions, conditions_names, 'loadblindness', dataframe, nb_trials, plot_args)

    # calculate the mean distribution and the credible interval
    print('finished')
