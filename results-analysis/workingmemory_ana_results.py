from utils import *
from pymc.data import change_accuracy_for_correct_column, convert_to_global_task
from utils import retrieve_and_init_models, add_difference_pre_post, get_pymc_trace


# ### UTILS FUNCTIONS FOR TREATING THE DATA ###
def format_data(path, save_lfa):
    task = "workingmemory"
    # FIRST TREAT THE CSV AND PARSE IT TO DF
    csv_path = f"{path}/{task}.csv"
    dataframe = pd.read_csv(csv_path, sep=",")
    # Conditions:
    number_condition = [4, 5, 6, 7, 8]
    # Few pre-processing
    dataframe = delete_uncomplete_participants(dataframe)
    dataframe["results_correct"] = dataframe.apply(lambda row: transform_string_to_row(row, "results_correct"),
                                                   axis=1)
    dataframe["results_num_stim"] = dataframe.apply(lambda row: transform_string_to_row(row, "results_num_stim"),
                                                    axis=1)
    # Other pre-processing (get accuracies and nb_correct
    for t in number_condition:
        dataframe[str(t)] = dataframe.apply(lambda row: compute_numbercond(row, t), axis=1)
        dataframe[f'{t}-correct'] = dataframe.apply(lambda row: compute_sum_to_row(row, str(t)), axis=1)
        dataframe[f'{t}-nb'] = dataframe.apply(lambda row: len(row[str(t)]), axis=1)
        dataframe[f'{t}-accuracy'] = dataframe[f'{t}-correct'] / dataframe[f'{t}-nb']
    dataframe['total-task-correct'] = convert_to_global_task(dataframe, [f'{col}-correct' for col in number_condition])
    dataframe['total-task-nb'] = 12 * len(number_condition)
    dataframe['total-task-accuracy'] = dataframe['total-task-correct'] / dataframe['total-task-nb']
    condition_accuracy_names = [f"{elt}-accuracy" for elt in number_condition] + ['total-task-accuracy']
    condition_correct_names = [f"{elt}-correct" for elt in number_condition] + ['total-task-correct']
    condition_nb_names = [f"{elt}-nb" for elt in number_condition] + ['total-task-nb']
    base = ['participant_id', 'task_status', 'condition']
    dataframe = dataframe[base + condition_accuracy_names + condition_correct_names + condition_nb_names]
    # If save_mode, store the dataframe into csv:
    if save_lfa:
        dataframe.to_csv(f'{path}/workingmemory_lfa.csv', index=False)
    return dataframe


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


# ## RUN FITTED MODELS AND PLOT VISUALISATIONS ###
def retrieve_zpdes_vs_baseline(study, conditions_to_keep, model_type, model=None):
    task = "workingmemory"
    path = f"../outputs/{study}/results_{study}/{task}"
    df = format_data(path, save_lfa=False)
    condition_list = ['4', '5', '6', '7', '8', 'total-task']
    columns = [f"{c}-accuracy" for c in condition_list] + [f"{c}-correct" for c in condition_list]
    df = pd.concat([df, add_difference_pre_post(df, columns)], axis=0)
    root_path = f"{study}-{model_type}"
    model_baseline = retrieve_and_init_models(root_path, task, conditions_to_keep, df, model, group="baseline")
    model_zpdes = retrieve_and_init_models(root_path, task, conditions_to_keep, df, model, group="zpdes")
    return model_zpdes, model_baseline


def run_visualisation(study, conditions_to_keep, model_type, model=None):
    model_zpdes, model_baseline = retrieve_zpdes_vs_baseline(study, conditions_to_keep, model_type, model)
    model_baseline.plot_posterior_and_population()
    model_baseline.plot_comparison_posterior_and_population(model_zpdes)


# ## FITTING MODELS:####
def fit_model(study, conditions_to_fit, model=None, model_type="pooled_model"):
    task = "workingmemory"
    path = f"../outputs/{study}/results_{study}/{task}"
    df = format_data(path, save_lfa=False)
    if model:
        get_pymc_trace(df, conditions_to_fit, task=task, model_object=model, model_type=model_type, study=study)


def get_stan_accuracy(dataframe):
    condition_names = [f'{elt}-accuracy' for elt in range(4, 9)]
    number_condition = [4, 5, 6, 7, 8]
    nb_trials = len(dataframe.loc[0, '4'])
    # For accuracy analysis, let's focus on the outcomes:
    for condition_name, condition in zip(condition_names, number_condition):
        dataframe[condition_name] = dataframe.apply(lambda row: np.mean(row[str(condition)]), axis=1)
    stan_distributions = get_stan_accuracy_distributions(dataframe, condition_names, nb_trials, 'workingmemory')
    # Draw figures for accuracy data
    plt_args = {'list_xlim': [-0.25, 4.25], 'list_ylim': [0, 1],
                'list_set_xticklabels': ['4', '5', '6', '7', '8'], 'list_set_xticks': [0, 1, 2, 3, 4],
                'list_set_yticklabels': ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                'list_set_yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                'scale_jitter': 0.5}
    plot_all_accuracy_figures(stan_distributions=stan_distributions, outcomes_names=condition_names,
                              task_name='workingmemory', overall_initial_data=dataframe, nb_trials=nb_trials,
                              plot_args=plt_args, study=study)


if __name__ == '__main__':
    path = "../outputs/v0_axa/results_v0_axa/workingmemory"
    study = "v0_axa"
    # run(study)
    # Keep this old version for some times:
    # extract observer index information
    # indices_id = extract_id(dataframe, num_count=2)
    # sumirize two days experiments
    # sum_observers = []
    # condition_names = [f'{elt}-accuracy' for elt in range(4, 9)]
    # # Each condition have same number of trials i.e 12
    # nb_trials = len(dataframe.loc[0, '4'])
    # for ob in indices_id:
    #     tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
    #     # Summation over the two sessions:
    #     sum_observers.append([ob] + [np.sum(tmp_df[f"sum-{col}"]) / (2 * nb_trials) for col in number_condition])
    # sum_observers = pd.DataFrame(sum_observers, columns=['participant_id'] + condition_names)
    # sum_observers['total_resp'] = sum_observers.apply(lambda row: 2 * nb_trials, axis=1)  # two days task
    # def format_for_pymc(df_wm):
    #     # # WORKING MEMORY # #
    #     # df_wm = pd.read_csv(os.path.join(path, "workingmemory_lfa.csv"))
    #     wm_cdt = ['4', '5', '6', '7', '8']
    #     df_wm = df_wm.rename(change_accuracy_for_correct_column, axis='columns')
    #     df_wm[[col for col in df_wm.columns if 'correct' in col]] = df_wm[[col for col in df_wm.columns if
    #                                                                        'correct' in col]] * 12
    #     # df_wm['total_resp'] = 12
    #     for cdt in wm_cdt:
    #         df_wm[cdt + '-nb'] = 12
    #     df_wm['total-task-correct'] = convert_to_global_task(df_wm, [col + '-correct' for col in wm_cdt])
    #     df_wm['total-task-nb'] = 12 * len(wm_cdt)
    #     wm_cdt.append('total-task')
    #     return df_wm, wm_cdt
    # def extract_mu_ci_from_summary_accuracy(dataframe, ind_cond):
    #     out = np.zeros((len(ind_cond), 3))  # 3 means the mu, ci_min, and ci_max
    #     for t, ind in enumerate(ind_cond):
    #         out[t, 0] = dataframe[ind].mu_theta
    #         out[t, 1] = dataframe[ind].ci_min
    #         out[t, 2] = dataframe[ind].ci_max
    #     return out
# def get_pymc_trace(data, condition_list, model_object, study, sample_size=4000):
#     model_baseline = model_object(data[data['condition'] == 'baseline'],
#                                   name='workingmemory', group='baseline', folder=f'{study}-pooled_model',
#                                   stim_cond_list=condition_list,
#                                   sample_size=sample_size)
#     model_baseline.run()
#     model_zpdes = model_object(data[data['condition'] == 'zpdes'],
#                                name='workingmemory', group='zpdes', folder=f'{study}-pooled_model',
#                                stim_cond_list=condition_list,
#                                sample_size=sample_size)
#     model_zpdes.run()
