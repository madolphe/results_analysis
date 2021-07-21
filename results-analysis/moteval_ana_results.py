import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from cal_stan_accuracy_rt import CalStan_accuracy, CalStan_rt


def compute_mean_per_condition(row):
    """
    3 conditions for MOT: speed=1,4 or 8
    Compute mean accuracy and mean RT for each condition
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


def count_number_of_trials(row):
    return len(row['results_correct'])


def compute_result_sum_hr(row):
    return 18 - row['result_nb_omission']


def extract_mu_ci_from_summary_accuracy(dataframe, ind_cond):
    out = np.zeros((len(ind_cond), 3))  # 3 means the mu, ci_min, and ci_max
    for t, ind in enumerate(ind_cond):
        out[t, 0] = dataframe[ind].mu_theta
        out[t, 1] = dataframe[ind].ci_min
        out[t, 2] = dataframe[ind].ci_max
    return out


def extract_mu_ci_from_summary_rt(dataframe):
    out = np.zeros((1, 3))  # 3 means the mu, ci_min, and ci_max
    out[0, 0] = dataframe.mu_rt
    out[0, 1] = dataframe.ci_min
    out[0, 2] = dataframe.ci_max
    return out


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
    axes.set_xticks([1, 2, 3])
    #axes.set_xlim([0.5, 2.5])
    axes.set_xticklabels(['1', '4', '8'], fontsize=20)
    axes.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=20)
    fig.savefig(fname_save)
    plt.show()


if __name__ == '__main__':
    csv_path = "../outputs/moteval/moteval.csv"
    dataframe = pd.read_csv(csv_path)
    dataframe = dataframe.apply(lambda row: transform_str_to_list(row, [
        'results_responses', 'results_rt', 'results_speed_stim', 'results_correct']), axis=1)
    dataframe = delete_uncomplete_participants(dataframe)
    dataframe = dataframe.apply(compute_mean_per_condition, axis=1)
    dataframe.to_csv('../outputs/moteval/moteval_treat.csv')

    # from here written by mswym
    # condition extraction
    # dataframe['result_correct'] = dataframe.apply(compute_result_sum_hr, axis=1)
    # dataframe['result_nb_omission']

    # extract observer index information
    indices_id = extract_id(dataframe, num_count=2)

    # sumirize two days experiments
    sum_observers = []
    outcomes_names = ["1-RT", "1-accuracy", "4-RT", "4-accuracy", "8-RT", "8-accuracy"]
    for ob in indices_id:
        print(ob)
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        sum_observers.append([np.mean(tmp_df[index]) for index in outcomes_names])

    sum_observers = pd.DataFrame(sum_observers, columns=outcomes_names)

    sum_observers['total_resp'] = dataframe.apply(count_number_of_trials, axis=1)  # two days task

    for col in outcomes_names:
        if 'accuracy' in col:
            sum_observers[col] = sum_observers[col] * sum_observers['total_resp']

    # for save summary data
    sum_observers.to_csv('../outputs/moteval/sumdata_moteval.csv')

    # calculate the mean distribution and the credible interval

    class_stan_accuracy = [CalStan_accuracy(sum_observers, ind_corr_resp=n) for n in
                           [f"{elt}-accuracy" for elt in [1, 4, 8]]]
    # class_stan_rt = CalStan_rt(sum_observers, ind_rt=2, max_rt=1000)

    # draw figures
    # for accuracy data
    dist_ind = sum_observers.loc[:, [f"{elt}-accuracy" for elt in [1, 4, 8]]].values / 45
    dist_summary = extract_mu_ci_from_summary_accuracy(class_stan_accuracy, [0, 1, 2])
    draw_all_distributions(dist_ind, dist_summary, len(sum_observers), num_cond=3, std_val=0.05,
                           fname_save='../outputs/moteval/moteval_hrfar.png')

    # dist_ind = sum_observers.iloc[0:len(sum_observers), 2].values
    # dist_summary = extract_mu_ci_from_summary_rt(class_stan_rt)
    # draw_all_distributions_rt(dist_ind, dist_summary, len(sum_observers), num_cond=1, std_val=0.05,
    #                           fname_save='../outputs/moteval/moteval_rt.png')
    print('finished')
