from utils import *


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
    # -------------------------------------------------------------------#
    # FIRST TREAT THE CSV AND PARSE IT TO DF
    # data loading
    csv_path = "../outputs/workingmemory/workingmemory.csv"
    dataframe = pd.read_csv(csv_path, sep=";")
    dataframe = delete_uncomplete_participants(dataframe)
    dataframe["results_correct"] = dataframe.apply(lambda row: transform_string_to_row(row, "results_correct"),
                                                   axis=1)
    dataframe["results_num_stim"] = dataframe.apply(lambda row: transform_string_to_row(row, "results_num_stim"),
                                                    axis=1)
    # -------------------------------------------------------------------#

    # -------------------------------------------------------------------#
    # THEN EXTRACT COLUMNS FOR FUTURE LATENT FACTOR ANALYSIS
    # condition extraction
    number_condition = [4, 5, 6, 7, 8]
    for t in number_condition:
        dataframe[str(t)] = dataframe.apply(lambda row: compute_numbercond(row, t), axis=1)
        dataframe['sum-' + str(t)] = dataframe.apply(lambda row: compute_sum_to_row(row, str(t)), axis=1)
        dataframe[f'{t}-nb'] = dataframe.apply(lambda row: len(row[str(t)]), axis=1)
        dataframe[f'{t}-accuracy'] = dataframe['sum-' + str(t)] / dataframe[f'{t}-nb']
    condition_accuracy_names = [f"{elt}-accuracy" for elt in number_condition]
    dataframe[['participant_id', 'task_status'] + condition_accuracy_names].to_csv(
        '../outputs/workingmemory/workingmemory_lfa.csv', index=False)
    # extract observer index information
    indices_id = extract_id(dataframe, num_count=2)
    # sumirize two days experiments
    sum_observers = []
    condition_names = [f'{elt}-accuracy' for elt in range(4, 9)]
    # Each condition have same number of trials i.e 12
    nb_trials = len(dataframe.loc[0, '4'])
    for ob in indices_id:
        print(ob)
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        # Summation over the two sessions:
        sum_observers.append([ob] + [np.sum(tmp_df[f"sum-{col}"]) / (2 * nb_trials) for col in number_condition])
        # sum_observers.append(
        #     [np.sum(tmp_df.sum4), np.sum(tmp_df.sum5), np.sum(tmp_df.sum6), np.sum(tmp_df.sum7), np.sum(tmp_df.sum8)])
    sum_observers = pd.DataFrame(sum_observers, columns=['participant_id'] + condition_names)
    sum_observers['total_resp'] = sum_observers.apply(lambda row: 2 * nb_trials, axis=1)  # two days task
    sum_observers.to_csv('../outputs/workingmemory/sumdata_workingmemory.csv', header=True, index=False)

    # -------------------------------------------------------------------#
    # BAYES ACCURACY ANALYSIS
    # For accuracy analysis, let's focus on the outcomes:
    for condition_name, condition in zip(condition_names, number_condition):
        dataframe[condition_name] = dataframe.apply(lambda row: np.mean(row[str(condition)]), axis=1)
    stan_distributions = get_stan_accuracy_distributions(dataframe, condition_names, nb_trials)
    # Draw figures for accuracy data
    plt_args = {'list_xlim': [0.75, 5.25], 'list_ylim': [0, 1],
                'list_set_xticklabels': ['4', '5', '6', '7', '8'], 'list_set_xticks': [1, 2, 3, 4, 5],
                'list_set_yticklabels': ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                'list_set_yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0]}
    plot_all_accuracy_figures(stan_distributions, condition_names, 'workingmemory', dataframe, nb_trials, plt_args)
    # -------------------------------------------------------------------#
    print('finished')
