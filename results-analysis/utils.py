import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


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


def draw_all_distributions(dist_ind,dist_summary,num_dist,num_cond,std_val = 0.05,list_xlim=[0.75,5.25],list_ylim=[0,1],
                                list_set_xticklabels=['5','6','7','8','9'],list_set_xticks=[1,2,3,4,5],
                                list_set_yticklabels=['0.0','0.2','0.4','0.6','0.8','1.0'],list_set_yticks=[0,0.2,0.4,0.6,0.8,1.0],
                                val_ticks = 0.02,
                                fname_save='workingmemory_accuracy.png'):
    # dist_ind is the matrix of (num_observers x num_conditions)
    # dist_summary is the mu (numb_conditions), ci_min, and ci_max
    x = [j+1+std_val*np.random.randn() for j in range(num_cond) for t in range(num_dist)]
    dist_ind = dist_ind.T.flatten()
    x_sum = np.linspace(1,num_cond,num_cond)
    fig = plt.figure(figsize=(5,5))
    axes = fig.add_subplot(1,1,1)
    axes.scatter(x,dist_ind,s=10,c='blue')
    axes.errorbar(x_sum, dist_summary[:,0], yerr = [dist_summary[:,0]-dist_summary[:,1], dist_summary[:,2]-dist_summary[:,0]], capsize=5, fmt='o', markersize=13, ecolor='red', markeredgecolor = "red", color='w')
    axes.set_xlim(list_xlim)
    axes.set_ylim(list_ylim)
    axes.xaxis.grid(True, which='minor')
    axes.set_xticks(list_set_xticks)
    axes.set_xticklabels(list_set_xticklabels,fontsize=20)
    axes.set_yticks(list_set_yticks)
    axes.set_yticklabels(list_set_yticklabels,fontsize=20)
    axes.grid()
    axes.yaxis.set_minor_locator(MultipleLocator(val_ticks))
    fig.savefig(fname_save)
    plt.show()

