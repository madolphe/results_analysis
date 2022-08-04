from utils import *
from pymc.data import change_accuracy_for_correct_column, convert_to_global_task
from utils import retrieve_and_init_models, add_difference_pre_post, get_pymc_trace


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


def format_data(path, save_lfa=False):
    csv_path = f"{path}/loadblindness.csv"
    conditions_names = ['near', 'far', 'total-task']
    dataframe = pd.read_csv(csv_path, sep=",")
    dataframe = delete_uncomplete_participants(dataframe)
    dataframe["results_responses_pos"] = dataframe.apply(
        lambda row: transform_string_to_row(row, "results_responses_pos"),
        axis=1)
    dataframe["results_target_distance"] = dataframe.apply(
        lambda row: transform_string_to_row(row, "results_target_distance"),
        axis=1)
    # For each condition:
    dataframe['far_response'] = dataframe.apply(lambda row: compute_nearfarcond(row, 1), axis=1)
    dataframe['near_response'] = dataframe.apply(lambda row: compute_nearfarcond(row, 0), axis=1)
    dataframe['far-correct'] = dataframe.apply(lambda row: compute_sum_to_row(row, "far_response"), axis=1)
    dataframe['near-correct'] = dataframe.apply(lambda row: compute_sum_to_row(row, "near_response"), axis=1)
    dataframe['near-nb'], dataframe['far-nb'] = dataframe['near_response'].apply(lambda row: len(row)), dataframe[
        'far_response'].apply(lambda row: len(row))
    dataframe['total_resp'] = dataframe.apply(lambda row: 20, axis=1)
    dataframe['near-accuracy'] = dataframe['near-correct'] / dataframe['near-nb']
    dataframe['far-accuracy'] = dataframe['far-correct'] / dataframe['far-nb']

    # Total task:
    dataframe['total-task-correct'] = dataframe['far-correct'] + dataframe['near-correct']
    dataframe['total-task-accuracy'] = (dataframe['near-accuracy'] + dataframe['far-accuracy']) / 2
    dataframe['total-task-nb'] = dataframe['near-nb'] + dataframe['far-nb']

    # nb_trials = len(dataframe['near_response'][0])
    # dataframe = dataframe[['participant_id', 'task_status', 'condition'] + conditions_names]
    if save_lfa:
        dataframe.to_csv(f'{path}/loadblindness_lfa.csv', index=False)
    return dataframe


# ## RUN FITTED MODELS AND PLOT VISUALISATIONS ###
def retrieve_zpdes_vs_baseline(study, conditions_to_keep, model_type, model=None):
    task = "loadblindness"
    path = f"../outputs/{study}/results_{study}/{task}"
    df = format_data(path)
    condition_list = ['near', 'far', 'total-task']
    base = ['participant_id', 'task_status', 'condition']
    columns_accuracy = [f"{c}-accuracy" for c in condition_list]
    columns_correct = [f"{c}-correct" for c in condition_list]
    df = df[base+columns_correct+columns_accuracy]
    df = pd.concat([df, add_difference_pre_post(df, columns_accuracy)], axis=0)
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
    task = "loadblindness"
    path = f"../outputs/{study}/results_{study}/{task}"
    df = format_data(path, save_lfa=False)
    if model:
        get_pymc_trace(df, conditions_to_fit, task=task, model_object=model, model_type=model_type, study=study)


def get_stan_accuracy(dataframe, conditions_names, nb_trials, study):
    stan_distributions = get_stan_accuracy_distributions(dataframe, conditions_names, nb_trials, 'loadblindness')
    # Draw figures for accuracy data
    plot_args = {'list_xlim': [-0.5, 1.5], 'list_ylim': [0, 1],
                 'list_set_xticklabels': ['Near', 'Far'], 'list_set_xticks': [0, 1],
                 'list_set_yticklabels': ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                 'list_set_yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                 'scale_jitter': 0.2}
    plot_all_accuracy_figures(stan_distributions=stan_distributions, outcomes_names=conditions_names,
                              task_name='loadblindness', overall_initial_data=dataframe, nb_trials=nb_trials,
                              plot_args=plot_args, study=study)


if __name__ == '__main__':
    study = "v0_axa"
    # run(study)
    # sumirize two days experiments
    # sum_observers = []
    # # extract observer index information
    # indices_id = extract_id(dataframe, num_count=2)
    # for ob in indices_id:
    #     tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
    #     sum_observers.append([ob] + [np.mean(tmp_df.accuracy_near), np.sum(tmp_df.accuracy_far)])
    # sum_observers = pd.DataFrame(sum_observers, columns=['participant_id'] + conditions_names)
    # # for save summary data
    # sum_observers['total_resp'] = sum_observers.apply(lambda row: 40, axis=1)  # two days task
    # sum_observers.to_csv(f'{path}/sumdata_loadblindness.csv', header=True, index=False)
    # def format_for_pymc(df_lb):
    #     # # LOADBLINDNESS # #
    #     # df_lb = pd.read_csv(os.path.join(path, "loadblindness_lfa.csv"))
    #     lb_cdt = ['near', 'far']
    #     df_lb[[col for col in df_lb.columns if 'accuracy' in col]] = df_lb[[col for col in df_lb.columns if
    #                                                                         'accuracy' in col]] * 20
    #     df_lb = df_lb.rename(columns={'accuracy_near': 'near-correct', 'accuracy_far': 'far-correct'})
    #     # df_lb['total_resp'] = 20
    #     for cdt in lb_cdt:
    #         df_lb[cdt + '-nb'] = 20
    #     df_lb['total-task-correct'] = convert_to_global_task(df_lb, [col + '-correct' for col in lb_cdt])
    #     df_lb['total-task-nb'] = 40
    #     lb_cdt.append('total-task')
    #     return df_lb, lb_cdt
# def get_pymc_trace(data, condition_list, model_object, study, sample_size=4000):
#     model_baseline = model_object(data[data['condition'] == 'baseline'],
#                                   name='loadblindness', group='baseline', folder=f'{study}-pooled_model',
#                                   stim_cond_list=condition_list,
#                                   sample_size=sample_size)
#     model_baseline.run()
#     model_zpdes = model_object(data[data['condition'] == 'zpdes'],
#                                name='loadblindness', group='zpdes', folder=f'{study}-pooled_model',
#                                stim_cond_list=condition_list,
#                                sample_size=sample_size)
#     model_zpdes.run()
    # df_load = dataframe[['participant_id', 'task_status', 'condition'] + conditions_names]
    # get_stan_accuracy(dataframe, conditions_names, nb_trials, study)