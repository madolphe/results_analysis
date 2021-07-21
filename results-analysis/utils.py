import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def transform_str_to_list(row, columns):
    for column in columns:
        row[column] = row[column].split(",")
    return row


def delete_uncomplete_participants(dataframe: pd.DataFrame) -> pd.DataFrame:
    """

    """
    mask = pd.DataFrame(dataframe.participant_id.value_counts() < 2)
    participants_to_delete = mask[mask['participant_id'] == True].index.tolist()
    for id in participants_to_delete:
        dataframe = dataframe[dataframe['participant_id'] != id]
    return dataframe


def extract_id(dataframe, num_count):
    mask = pd.DataFrame(dataframe.participant_id.value_counts() == num_count)
    indices_id = mask[mask['participant_id'] == True].index.tolist()
    return indices_id


def draw_all_distributions(dist_ind, dist_summary, num_dist, num_cond, std_val=0.05,
                           fname_save='workingmemory_accuracy.png'):
    # dist_ind is the matrix of (num_observers x num_conditions)
    # dist_summary is the mu (numb_conditions), ci_min, and ci_max
    x = [j + 1 + std_val * np.random.randn() for j in range(num_cond) for t in range(num_dist)]
    dist_ind = dist_ind.T.flatten()
    x_sum = np.linspace(1, num_cond, num_cond)
    fig = plt.figure(figsize=(5, 5))
    axes = fig.add_subplot(1, 1, 1)
    axes.scatter(x, dist_ind, s=10, c='blue')
    axes.errorbar(x_sum, dist_summary[:, 0],
                  yerr=[dist_summary[:, 0] - dist_summary[:, 1], dist_summary[:, 2] - dist_summary[:, 0]], capsize=5,
                  fmt='o', markersize=15, ecolor='red', markeredgecolor="red", color='w')
    axes.set_xticks([1, 2])
    axes.set_xlim([0.5, 2.5])
    axes.set_xticklabels(['HR', 'FAR'], fontsize=20)
    axes.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=20)
    fig.savefig(fname_save)
    plt.show()


def draw_all_distributions_rt(dist_ind, dist_summary, num_dist, num_cond, std_val=0.05,
                              fname_save='memorability_rt.png'):
    # dist_ind is the matrix of (num_observers x num_conditions)
    # dist_summary is the mu (numb_conditions), ci_min, and ci_max
    x = [j + 1 + std_val * np.random.randn() for j in range(num_cond) for t in range(num_dist)]
    dist_ind = dist_ind.T.flatten()
    x_sum = np.linspace(1, num_cond, num_cond)
    fig = plt.figure(figsize=(5, 5))
    axes = fig.add_subplot(1, 1, 1)
    axes.scatter(x, dist_ind, s=10, c='blue')
    axes.errorbar(x_sum, dist_summary[:, 0],
                  yerr=[dist_summary[:, 0] - dist_summary[:, 1], dist_summary[:, 2] - dist_summary[:, 0]], capsize=5,
                  fmt='o', markersize=15, ecolor='red', markeredgecolor="red", color='w')
    axes.set_xlim([0.5, 1.5])
    axes.set_xticks([1])
    axes.set_xticklabels(['HR'], fontsize=20)
    axes.set_yticks([0, 250, 500, 750])
    axes.set_yticklabels(['0', '250', '500', '750'], fontsize=20)
    fig.savefig(fname_save)
    plt.show()
