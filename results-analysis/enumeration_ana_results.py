import pandas as pd

from utils import *
from pymc.data import change_accuracy_for_correct_column, convert_to_global_task
from utils import retrieve_and_init_models, add_difference_pre_post, get_pymc_trace


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


def format_data(path, save_lfa=False):
    # FIRST TREAT THE CSV AND PARSE IT TO DF
    task = "enumeration"
    # csv_path = f"../outputs/{study}/results_{study}/{task}/{task}.csv"
    df = pd.read_csv(f"{path}/{task}.csv", sep=",")
    conditions = ["5", "6", "7", "8", "9"]
    df['result_response_exact'] = df.apply(compute_result_exact_answers, axis=1)
    df['mean_rt_session'] = df.apply(compute_mean_per_row, axis=1)
    df['std_rt_session'] = df.apply(compute_std_per_row, axis=1)
    # pre_response_exact = dataframe[dataframe['task_status'] == "PRE_TEST"]['result_response_exact'].values
    # post_response_exact = dataframe[dataframe['task_status'] == "POST_TEST"]['result_response_exact'].values
    # condition extraction - add to dataframe a column result_correct where each cell is a list of 0 - 1
    # (binary success for each trial)
    df['result_correct'] = df.apply(compute_result_exact_answers_list, axis=1)
    base = ['participant_id', 'task_status', 'condition']
    condition_accuracy = [f"{i}-accuracy" for i in conditions]
    condition_correct = [f"{i}-correct" for i in conditions]
    condition_nb = [f"{i}-nb" for i in conditions]
    # Let's sort the 'result_correct' column by condition:
    # For each condition we create a list of 0-1 binary success
    # And we compute the number of success for each condition in the column {condition}-sum:
    for condition in conditions:
        df[f"{condition}-results"] = df.apply(lambda row: compute_numbercond(row, int(condition)), axis=1)
        df[f"{condition}-nb"] = df.apply(lambda row: len(row[f"{condition}-results"]), axis=1)
        df[f"{condition}-correct"] = df.apply(lambda row: np.sum(row[condition + "-results"]), axis=1)
        df[f"{condition}-accuracy"] = df[f"{condition}-correct"] / df[f"{condition}-nb"]
    condition_accuracy.append("total-task-accuracy")
    condition_correct.append("total-task-correct")
    df['total-task-correct'] = convert_to_global_task(df, [f'{cdt}-correct' for cdt in conditions])
    df['total-task-nb'] = 20 * len(conditions)
    df['total-task-accuracy'] = df['total-task-correct'] / df['total-task-nb']
    df = df[base + condition_correct + condition_accuracy + condition_nb]
    df = delete_uncomplete_participants(df)
    return df


# ## RUN FITTED MODELS AND PLOT VISUALISATIONS ###
def retrieve_zpdes_vs_baseline(study, conditions_to_keep, model_type, model=None):
    task = "enumeration"
    path = f"../outputs/{study}/results_{study}/{task}"
    root_path = f"{study}-{model_type}"
    df = format_data(path)
    condition_list = ["5", "6", "7", "8", "9"]
    columns = [f"{c}-accuracy" for c in condition_list] + [f"{c}-correct" for c in condition_list]
    df = pd.concat([df, add_difference_pre_post(df, columns)], axis=0)
    model_baseline = retrieve_and_init_models(root_path, task, conditions_to_keep, df, model, group="baseline")
    model_zpdes = retrieve_and_init_models(root_path, task, conditions_to_keep, df, model, group="zpdes")
    return model_zpdes, model_baseline


def run_visualisation(study, conditions_to_keep, model_type, model=None):
    model_zpdes, model_baseline = retrieve_zpdes_vs_baseline(study, conditions_to_keep, model_type, model)
    model_baseline.plot_posterior_and_population()
    model_baseline.plot_comparison_posterior_and_population(model_zpdes)


# ## FITTING MODELS:####
def fit_model(study, conditions_to_fit, model=None, model_type="pooled_model"):
    task = "enumeration"
    path = f"../outputs/{study}/results_{study}/{task}"
    df = format_data(path, save_lfa=False)
    if model:
        get_pymc_trace(df, conditions_to_fit, task=task, model_object=model, model_type=model_type, study=study)


def get_stan_accuracy(dataframe, condition_names, study):
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
    plot_all_accuracy_figures(stan_distributions=stan_distributions, outcomes_names=condition_names,
                              task_name='enumeration', overall_initial_data=dataframe, nb_trials=nb_trials,
                              plot_args=plot_args, study=study)
    print('finished')


if __name__ == '__main__':
    # study = 'v0_axa'
    # run(study)
    pass
# def plot_traces(study, model_type, model=None):
#     root_path = f"{study}-{model_type}"
#     dataframe, pre_response_exact, post_response_exact, condition_names, condition_possibilities = format_data(study)
#     df_enum = get_lfa_csv(dataframe, condition_possibilities, condition_names, study, save=False)
#     data, condition_list = format_for_pymc(df_enum)
#     dict_traces_baseline = {
#         condition: f"{root_path}/enumeration/enumeration_baseline_results/enumeration_baseline-{condition}-trace" for
#         condition in condition_list}
#     dict_traces_zpdes = {
#         condition: f"{root_path}/enumeration/enumeration_zpdes_results/enumeration_zpdes-{condition}-trace" for
#         condition in condition_list}
#     model_baseline = model(data[data['condition'] == 'baseline'],
#                            name='enumeration', group='baseline', folder=root_path,
#                            stim_cond_list=condition_list, traces_path=dict_traces_baseline,
#                            sample_size=4000)
#     model_zpdes = model(data[data['condition'] == 'zpdes'],
#                         name='enumeration', group='zpdes', folder=root_path,
#                         stim_cond_list=condition_list, traces_path=dict_traces_zpdes,
#                         sample_size=4000)
#     # model_baseline.plot_posterior_and_population()
#     model_baseline.plot_comparison_posterior_and_population(model_zpdes)

# def get_lfa_csv(dataframe, condition_possibilities, condition_names, study, save=True):
#     # Latent factor analysis:
#     # from here written by mswym
#     for index, condition in zip(condition_possibilities, condition_names):
#         print(index, condition, dataframe[str(index)].apply(lambda row: len(row))[0])
#         dataframe[f"{condition}"] = dataframe[f"{index}-sum"] / dataframe[str(index)].apply(lambda row: len(row))
#     # dataframe[['participant_id', 'task_status', 'condition'] + condition_names].to_csv(
#     #     '../outputs/v1_ubx/enumeration_lfa_v1.csv',
#     #     index=False)
#     if save:
#         dataframe[['participant_id', 'task_status', 'condition'] + condition_names].to_csv(
#             f'../outputs/{study}/enumeration_lfa_v1.csv',
#             index=False)
#     # summarize two days experiments
#     # sum_observers = []
#     # extract observer index information
#     # indices_id = extract_id(dataframe, num_count=2)
#     # for ob in indices_id:
#     #     tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
#     #     sum_observers.append([ob] + [tmp_df[col].mean(axis=0) for col in tmp_df.columns if "accuracy" in col])
#     # sum_observers = pd.DataFrame(sum_observers, columns=['participant_id'] + condition_names)
#     # sum_observers.to_csv('../outputs/enumeration/sumdata_enumeration.csv', header=True, index=False)
#     return dataframe[['participant_id', 'task_status', 'condition'] + condition_names]


# def format_for_pymc(df_enum):
#     # # ENUMERATION  # #
#     # Keep the accuracy and the total nb of correct elements to plot the population easily afterward
#     # df_enum = pd.read_csv(os.path.join(path, "enumeration_lfa_v1.csv"))
#     df = df_enum
#     df_enum = df_enum.rename(change_accuracy_for_correct_column, axis='columns')
#     df_enum[[col for col in df_enum.columns if 'correct' in col]] = df_enum[[col for col in df_enum.columns if
#                                                                              'correct' in col]] * 20
#     # Keep accuracy in the columns (for plotting purposes)
#     df_enum = pd.concat([df_enum, df[[col for col in df.columns if 'accuracy' in col]]], axis=1)
#     enum_cdt = ['5', '6', '7', '8', '9']
#     for cdt in enum_cdt:
#         df_enum[cdt + '-nb'] = 20
#     # df_enum['total_resp'] = 20
#     df_enum['total-task-correct'] = convert_to_global_task(df_enum, [col + '-correct' for col in enum_cdt])
#     df_enum['total-task-nb'] = 20 * len(enum_cdt)
#     df_enum = df_enum.groupby('participant_id').filter(lambda x: len(x) > 1)
#     enum_cdt.append('total-task')
#     return df_enum, enum_cdt
# def get_pymc_trace(data, condition_list, model_object, study, sample_size=4000):
#     model_baseline = model_object(data[data['condition'] == 'baseline'],
#                                   name='enumeration', group='baseline', folder=f'{study}-pooled_model',
#                                   stim_cond_list=condition_list,
#                                   sample_size=sample_size)
#     model_baseline.run()
#     model_zpdes = model_object(data[data['condition'] == 'zpdes'],
#                                name='enumeration', group='zpdes', folder=f'{study}-pooled_model',
#                                stim_cond_list=condition_list,
#                                sample_size=sample_size)
#     model_zpdes.run()
# -------------------------------------------------------------------#
# BAYES ACCURACY ANALYSIS w pystan
# get_stan_accuracy(dataframe[dataframe['condition'] == 'zpdes'], condition_names, study=study)
# -------------------------------------------------------------------#
