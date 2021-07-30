import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from cal_stan_accuracy_rt import CalStan_accuracy, CalStan_rt


def delete_uncomplete_participants(dataframe):
    """

    """
    mask = pd.DataFrame(dataframe.participant_id.value_counts() < 2)
    participants_to_delete = mask[mask['participant_id'] == True].index.tolist()
    for id in participants_to_delete:
        dataframe = dataframe[dataframe['participant_id'] != id]
    return dataframe


def transform_string_to_row(row, column):
    return [int(elt) for elt in row[column].split(',') if elt]


def compute_numbercond(row, ind_cond):
    """
        From the row of results, return the list of resuluts_correct according to results_num_stim condition 
    """
    results_responses = list(row["results_correct"])
    results_targetvalue = list(row["results_num_stim"])
    out = []
    out = [results_responses[ind] for ind, t in enumerate(results_targetvalue) if t == ind_cond]

    return np.array(out)


def compute_sum_to_row(row, column):
    return np.sum(row[column])


def extract_id(dataframe, num_count):
    mask = pd.DataFrame(dataframe.participant_id.value_counts() == num_count)
    indices_id = mask[mask['participant_id'] == True].index.tolist()
    return indices_id


def extract_mu_ci_from_summary_accuracy(dataframe, ind_cond):
    out = np.zeros((len(ind_cond), 3))  # 3 means the mu, ci_min, and ci_max
    for t, ind in enumerate(ind_cond):
        out[t, 0] = dataframe[ind].mu_theta
        out[t, 1] = dataframe[ind].ci_min
        out[t, 2] = dataframe[ind].ci_max
    return out


if __name__ == '__main__':
    # data loading
    csv_path = "../outputs/workingmemory/workingmemory.csv"
    dataframe = pd.read_csv(csv_path)
    dataframe = delete_uncomplete_participants(dataframe)
    dataframe["results_correct"] = dataframe.apply(lambda row: transform_string_to_row(row, "results_correct"),
                                                   axis=1)
    dataframe["results_num_stim"] = dataframe.apply(lambda row: transform_string_to_row(row, "results_num_stim"),
                                                    axis=1)

    # condition extraction
    number_condition = [4, 5, 6, 7, 8]
    for t in number_condition:
        dataframe[str(t)] = dataframe.apply(lambda row: compute_numbercond(row, t), axis=1)
        dataframe['sum' + str(t)] = dataframe.apply(lambda row: compute_sum_to_row(row, str(t)), axis=1)

    # extract observer index information
    indices_id = extract_id(dataframe, num_count=2)
    # sumirize two days experiments
    sum_observers = []
    condition_names = [f'{elt}-accuracy' for elt in range(4, 9)]
    for ob in indices_id:
        print(ob)
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        sum_observers.append(
            [np.sum(tmp_df.sum4), np.sum(tmp_df.sum5), np.sum(tmp_df.sum6), np.sum(tmp_df.sum7), np.sum(tmp_df.sum8)])
    sum_observers = pd.DataFrame(sum_observers)
    nb_trials = 24
    # for save summary data
    tmp = sum_observers / 24.
    tmp.to_csv('sumdata_workingmemory.csv', header=False, index=False)
    sum_observers['total_resp'] = sum_observers.apply(lambda row: 24, axis=1)  # two days task

    for condition in condition_names:
        dataframe[condition] = dataframe.apply(lambda row: np.mean(row[condition.replace("-accuracy", "")]), axis=1)

    stan_distributions = get_stan_accuracy_distributions(dataframe, condition_names, nb_trials)
    # Draw figures for accuracy data
    plt_args = {'list_xlim': [0.75, 5.25], 'list_ylim': [0, 1],
                'list_set_xticklabels': ['4', '5', '6', '7', '8'], 'list_set_xticks': [1, 2, 3, 4, 5],
                'list_set_yticklabels': ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                'list_set_yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0]}
    plot_all_accuracy_figures(stan_distributions, condition_names, 'workingmemory', dataframe, nb_trials, plt_args)

    # calculate the mean distribution and the credible interval
    # class_stan_accuracy = [CalStan_accuracy(sum_observers, ind_corr_resp=n) for n in range(5)]
    # draw figures
    # for accuracy data
    # dist_ind = sum_observers.iloc[0:len(sum_observers), 0:5].values / 24.
    # dist_summary = extract_mu_ci_from_summary_accuracy(class_stan_accuracy, [0, 1, 2, 3, 4])
    # draw_all_distributions(dist_ind, dist_summary, len(sum_observers), num_cond=5, std_val=0.05,
    #                        list_xlim=[0.75, 5.25], list_ylim=[0, 1],
    #                        list_set_xticklabels=['4', '5', '6', '7', '8'], list_set_xticks=[1, 2, 3, 4, 5],
    #                        list_set_yticklabels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
    #                        list_set_yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #                        fname_save='../outputs/workingmemory/workingmemory_accuracy.png')

    print('finished')
