from utils import *


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
        row[f"{key}-rt"] = np.mean(dict_mean_rt_per_condition[key])
        row[f"{key}-accuracy"] = np.mean(dict_mean_accuracy_per_condition[key])
        row[f"{key}-nb"] = len(dict_mean_accuracy_per_condition[key])
    return row


def count_number_of_trials(row):
    return len(row['results_correct'])


def compute_result_sum_hr(row):
    return 18 - row['result_nb_omission']


if __name__ == '__main__':
    # -------------------------------------------------------------------#
    # FIRST TREAT THE CSV AND PARSE IT TO DF
    csv_path = "../outputs/moteval/moteval.csv"
    dataframe = pd.read_csv(csv_path)
    dataframe = dataframe.apply(lambda row: transform_str_to_list(row, [
        'results_responses', 'results_rt', 'results_speed_stim', 'results_correct']), axis=1)
    dataframe = delete_uncomplete_participants(dataframe)
    dataframe = dataframe.apply(compute_mean_per_condition, axis=1)
    dataframe.to_csv('../outputs/moteval/moteval_treat.csv')
    # -------------------------------------------------------------------#

    # -------------------------------------------------------------------#
    # THEN EXTRACT COLUMNS FOR FUTURE LATENT FACTOR ANALYSIS
    # extract observer index information
    indices_id = extract_id(dataframe, num_count=2)
    # summarize two days experiments for Latent Factor Analysis
    sum_observers = []
    outcomes_names = ["1-rt", "1-accuracy", "4-rt", "4-accuracy", "8-rt", "8-accuracy"]
    for ob in indices_id:
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        sum_observers.append([np.mean(tmp_df[index]) for index in outcomes_names])
    sum_observers = pd.DataFrame(sum_observers, columns=outcomes_names)
    sum_observers['total_resp'] = dataframe.apply(count_number_of_trials, axis=1)  # two days task
    # for save summary data
    sum_observers.to_csv('../outputs/moteval/sumdata_moteval.csv')
    # -------------------------------------------------------------------#

    # -------------------------------------------------------------------#
    # BAYES ACCURACY ANALYSIS
    # For accuracy analysis, let's focus on the outcomes:
    outcomes_names = ["1-accuracy", "4-accuracy", "8-accuracy"]
    nb_trials = len(dataframe['results_correct'][0])
    stan_distributions = get_stan_accuracy_distributions(dataframe, outcomes_names, nb_trials)
    # Draw figures for accuracy data
    plt_args = {'list_xlim': [0.75, 3.25], 'list_ylim': [0, 1],
                'list_set_xticklabels': ['1', '4', '8'], 'list_set_xticks': [1, 2, 3],
                'list_set_yticklabels': ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                'list_set_yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0] }
    plot_all_accuracy_figures(stan_distributions, outcomes_names, 'moteval', dataframe, nb_trials, plt_args)
    # -------------------------------------------------------------------#

    # -------------------------------------------------------------------#
    # BAYES RT ANALYSIS:
    outcomes_names = ["1-rt", "4-rt", "8-rt"]
    stan_rt_distributions = get_stan_RT_distributions(dataframe, ["1", "4", "8"])
    plt_args = {'list_xlim': [0.75, 3.25], 'list_ylim': [0, 1],
                'list_set_xticklabels': ['1', '4', '8'], 'list_set_xticks': [1, 2, 3],
                'list_set_yticklabels': ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                'list_set_yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0]}
    plot_all_rt_figures(stan_rt_distributions, outcomes_names, dataframe=dataframe, task_name='moteval',
                        plot_args=plt_args)
