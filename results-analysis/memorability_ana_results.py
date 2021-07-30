import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from extract_sorted_memory import Results_memory
from utils import *
from cal_stan_accuracy_rt import CalStan_accuracy, CalStan_rt


# Treat data:
def compute_result_exact_answers(row):
    response = row["results_responses"].split(',')
    target = row["results_targetvalue"].split(',')

    return sum(x == y for x, y in zip(response, target))


def delete_uncomplete_participants(dataframe):
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


def extract_mu_ci_from_summary_accuracy(dataframe, ind_cond):
    outs = np.zeros((len(ind_cond), 3))  # 3 means the mu, ci_min, and ci_max
    for t, ind in enumerate(ind_cond):
        outs[t, 0] = dataframe[ind].mu_theta
        outs[t, 1] = dataframe[ind].ci_min
        outs[t, 2] = dataframe[ind].ci_max
    return outs


def extract_mu_ci_from_summary_rt(dataframe, ind_cond):
    outs = np.zeros((len(ind_cond), 3))  # 3 means the mu, ci_min, and ci_max
    for t, ind in enumerate(ind_cond):
        outs[t, 0] = dataframe[ind].mu_rt
        outs[t, 1] = dataframe[ind].ci_min
        outs[t, 2] = dataframe[ind].ci_max
    return outs


if __name__ == '__main__':

    csv_path_1 = "../outputs/memorability/memorability_1.csv"
    dataframe_1 = pd.read_csv(csv_path_1)
    dataframe_1 = delete_uncomplete_participants(dataframe_1)
    dataframe_1['session'] = 1

    csv_path_2 = "../outputs/memorability/memorability_2.csv"
    dataframe_2 = pd.read_csv(csv_path_2)
    dataframe_2 = delete_uncomplete_participants(dataframe_2)
    dataframe_2['session'] = 2

    dataframe = pd.concat([dataframe_1, dataframe_2], axis=0)

    # Let's create the condition names:
    test_status = ["PRE_TEST", "POST_TEST"]
    conditions = [*[f"{elt}" for elt in range(2, 6)], "100"]
    conditions_names_hit_miss = [f"{elt}-hit-miss" for elt in conditions]
    conditions_names_fa_cr = [f"{elt}-fa-cr" for elt in conditions]
    conditions_names_rt = [f"{elt}-rt" for elt in conditions]
    conditions_names = [conditions_names_hit_miss, conditions_names_fa_cr, conditions_names_rt]
    dataframe[conditions_names_hit_miss + conditions_names_fa_cr + conditions_names_rt] = None
    sum_observers = []
    tmp_overall_results = []
    indices_id = extract_id(dataframe, num_count=4)

    for ob in indices_id:
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        tmp_results = Results_memory(tmp_df)
        sum_observers.append(np.concatenate((tmp_results.out_mat_hit_miss_sum, tmp_results.out_mat_fa_cr_sum,
                                             tmp_results.out_mat_rt_cond, tmp_results.out_mat_rt_cond_std)))
        # Task status index varies from 0 to 3 => pre/post for memo 1 and 2
        # We use Results_memory for one participant_id for pre and post (and we do it for both memo 1 and 2)
        # We don't need to update the second part of memorability dataframe BUT this part is needed in Results_memory()
        # initialisation as it's the long range condition
        for task_status_index, (row_index, row) in enumerate(dataframe[dataframe['participant_id'] == ob].iterrows()):
            tmp_row = Results_memory(tmp_df[tmp_df['task_status'] == test_status[task_status_index % 2]])
            for conditions in conditions_names:
                for condition_index, condition in enumerate(conditions):
                    if 'hit' in condition:
                        tmp_cond = 'out_mat_hit_miss_sum'
                    elif 'fa' in condition:
                        tmp_cond = 'out_mat_fa_cr_sum'
                    else:
                        tmp_cond = 'out_mat_rt_cond'
                    dataframe.loc[row_index, condition] = tmp_row.__dict__[tmp_cond][condition_index]

    sum_observers = pd.DataFrame(sum_observers)
    # for save summary data
    # tmp = sum_observers/32.
    # tmp.loc[:,10:20] = tmp.loc[:,10:20]*32
    # tmp.to_csv('sumdata_memorability.csv',header=False, index=False)
    sum_observers['total_resp'] = sum_observers.apply(lambda row: 32, axis=1)
    dataframe['total_resp'] = sum_observers.apply(lambda row: 32, axis=1)
    # import pdb;pdb.set_trace()

    # for hr data
    # -------------------------------------------------------------------#
    # BAYES ACCURACY ANALYSIS
    # For accuracy analysis, let's focus on the outcomes:
    # Just drop second part of df that is useless:
    dataframe = dataframe[dataframe['session'] == 0]
    nb_trials = 32
    dataframe[conditions_names_hit_miss] = dataframe[conditions_names_hit_miss] / nb_trials
    stan_distributions = get_stan_accuracy_distributions(dataframe, conditions_names_hit_miss, nb_trials)
    # Draw figures for accuracy data
    plt_args = {'list_xlim': [0.75, 5.25], 'list_ylim': [0, 1],
                'list_set_xticklabels': ['2', '3', '4', '5', '>100'], 'list_set_xticks': [1, 2, 3, 4, 5],
                'list_set_yticklabels': ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                'list_set_yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0]}
    plot_all_accuracy_figures(stan_distributions, conditions_names_hit_miss, 'memorability', dataframe, nb_trials,
                              plt_args)

    # -------------------------------------------------------------------#
    # class_stan_accuracy = [CalStan_accuracy(sum_observers, ind_corr_resp=n) for n in range(10)]
    # dist_ind = sum_observers.iloc[0:len(sum_observers), 0:5].values / 32.
    # dist_summary = extract_mu_ci_from_summary_accuracy(class_stan_accuracy, [0, 1, 2, 3, 4])

    # class_stan_rt = [CalStan_rt(sum_observers, ind_rt=10 + n, max_rt=1400) for n in range(5)]
    # for far data
    # dist_ind = sum_observers.iloc[0:len(sum_observers), 5:10].values / 32.
    # dist_summary = extract_mu_ci_from_summary_accuracy(class_stan_accuracy, [5, 6, 7, 8, 9])
    # draw_all_distributions(dist_ind, dist_summary, len(sum_observers), num_cond=5, std_val=0.05,
    #                        list_xlim=[0.75, 5.25], list_ylim=[0, 1],
    #                        list_set_xticklabels=['2', '3', '4', '5', '>100'], list_set_xticks=[1, 2, 3, 4, 5],
    #                        list_set_yticklabels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
    #                        list_set_yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #                        fname_save='../outputs/memorability/memorability_far.png')

    # dist_ind = sum_observers.iloc[0:len(sum_observers), 10:15].values
    # dist_summary = extract_mu_ci_from_summary_rt(class_stan_rt, [0, 1, 2, 3, 4])
    # draw_all_distributions(dist_ind, dist_summary, len(sum_observers), num_cond=5, std_val=0.05,
    #                        list_xlim=[0.75, 5.25], list_ylim=[0, 1200],
    #                        list_set_xticklabels=['2', '3', '4', '5', '>100'], list_set_xticks=[1, 2, 3, 4, 5],
    #                        list_set_yticklabels=['0', '400', '800', '1200'], list_set_yticks=[0, 400, 800, 1200],
    #                        val_ticks=25,
    #                        fname_save='../outputs/memorability/memorability_rt.png')

    print('finished')
