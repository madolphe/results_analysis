from utils import *


# Treat data:
def compute_result_exact_answers(row):
    response = row["results_responses"].split(',')
    target = row["results_targetvalue"].split(',')
    return sum(x == y for x, y in zip(response, target))


def compute_mean_per_row(row):
    return_value = np.array(row['results_rt'].split(','), dtype=np.int32)
    return np.mean(return_value)


def compute_std_per_row(row):
    return_value = np.array(row['results_rt'].split(','), dtype=np.int32)
    return np.std(return_value)


def compute_result_exact_answers_list(row):
    """
    Returns a binary sucess for each trial
    """
    response = row["results_responses"].split(',')
    target = row["results_targetvalue"].split(',')
    out = [1 if x == y else 0 for x, y in zip(response, target)]
    return np.array(out)


def compute_numbercond(row, ind_cond):
    """
        From the row of results, return the list of resuluts_correct according to results_num_stim condition 
    """
    results_responses = list(row["result_correct"])
    results_targetvalue = [int(t) for t in row["results_targetvalue"].split(',')]
    out = [results_responses[ind] for ind, t in enumerate(results_targetvalue) if t == ind_cond]
    return np.array(out)


def plt_reliability_freq(pre_response_exact, post_response_exact):
    # Reliability of measurement
    pearson_coeff = np.corrcoef(pre_response_exact, post_response_exact)[1, 0] ** 2
    plt.scatter(pre_response_exact, post_response_exact)
    plt.title(f"Pearson coefficient: {pearson_coeff}")


def plt_RT():
    # Mean and SD reaction time plots and values:
    fig, axs = plt.subplots(1, len(dataframe.task_status.unique()), figsize=(10, 5), sharey=False)
    boxplot = dataframe.boxplot(column=['mean_rt_session', 'result_response_exact'], by=['task_status'], layout=(2, 1),
                                ax=axs)
    # plt.show()


def format_data():
    # FIRST TREAT THE CSV AND PARSE IT TO DF
    # csv_path = "../outputs/enumeration/enumeration.csv"
    csv_path = "../outputs/v1_ubx/results_v1_ubx/enumeration.csv"
    dataframe = pd.read_csv(csv_path, sep=",")
    dataframe['result_response_exact'] = dataframe.apply(compute_result_exact_answers, axis=1)
    dataframe['mean_rt_session'] = dataframe.apply(compute_mean_per_row, axis=1)
    dataframe['std_rt_session'] = dataframe.apply(compute_std_per_row, axis=1)
    pre_response_exact = dataframe[dataframe['task_status'] == "PRE_TEST"]['result_response_exact'].values
    post_response_exact = dataframe[dataframe['task_status'] == "POST_TEST"]['result_response_exact'].values
    # condition extraction - add to dataframe a column result_correct where each cell is a list of 0 - 1
    # (binary success for each trial)
    dataframe['result_correct'] = dataframe.apply(compute_result_exact_answers_list, axis=1)
    # Number of targets goes from 5 to 9:
    condition_possibilities = [i for i in range(5, 10)]
    # Let's sort the 'result_correct' column by condition:
    # For each condition we create a list of 0-1 binary success
    # And we compute the number of success for each condition in the column {condition}-sum:
    for condition in condition_possibilities:
        dataframe[f"{condition}"] = dataframe.apply(lambda row: compute_numbercond(row, condition), axis=1)
        dataframe[f"{condition}-sum"] = dataframe.apply(lambda row: np.sum(row[str(condition)]), axis=1)
    # Lets compute accuracy:
    condition_names = [f"{i}-accuracy" for i in condition_possibilities]
    return dataframe, pre_response_exact, post_response_exact, condition_names, condition_possibilities


def get_lfa_csv(dataframe, condition_possibilities, condition_names):
    # Latent factor analysis:
    # from here written by mswym
    for index, condition in zip(condition_possibilities, condition_names):
        print(index, condition, dataframe[str(index)].apply(lambda row: len(row))[0])
        dataframe[f"{condition}"] = dataframe[f"{index}-sum"] / dataframe[str(index)].apply(lambda row: len(row))
    dataframe[['participant_id', 'task_status', 'condition'] + condition_names].to_csv(
        '../outputs/v1_ubx/enumeration_lfa_v1.csv',
        index=False)
    # summarize two days experiments
    sum_observers = []
    # extract observer index information
    indices_id = extract_id(dataframe, num_count=2)
    for ob in indices_id:
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        sum_observers.append([ob] + [tmp_df[col].mean(axis=0) for col in tmp_df.columns if "accuracy" in col])
    sum_observers = pd.DataFrame(sum_observers, columns=['participant_id'] + condition_names)
    # sum_observers.to_csv('../outputs/enumeration/sumdata_enumeration.csv', header=True, index=False)
    return


def get_stan_accuracy(dataframe, condition_names):
    # nb_trials = 20 per condition:
    nb_trials = len(dataframe['5'][0])
    stan_distributions = get_stan_accuracy_distributions(dataframe, condition_names, nb_trials, 'enumeration')
    # Draw figures for accuracy data
    plot_args = {'list_xlim': [-0.25, 4.25], 'list_ylim': [0, 1],
                 'list_set_xticklabels': [str(i) for i in range(5, 10)],
                 'list_set_xticks': [i for i in range(0, 5)],
                 'list_set_yticklabels': ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                 'list_set_yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                 'scale_jitter': 0.5}
    plot_all_accuracy_figures(stan_distributions, condition_names, 'enumeration', dataframe, nb_trials, plot_args)
    print('finished')


if __name__ == '__main__':
    # -------------------------------------------------------------------#
    # FORMAT ALL DATA:
    dataframe, pre_response_exact, post_response_exact, condition_names, condition_possibilities = format_data()
    # -------------------------------------------------------------------#
    # GET CSV FOR ANALYSIS
    get_lfa_csv(dataframe, condition_possibilities, condition_names)
    # -------------------------------------------------------------------#
    # BAYES ACCURACY ANALYSIS
    # get_stan_accuracy(dataframe, condition_names)
