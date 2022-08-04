from utils import *
from pymc.data import change_accuracy_for_correct_column, convert_to_global_task
from utils import retrieve_and_init_models, add_difference_pre_post, get_pymc_trace


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


def format_data(path, save_lfa=False):
    # FIRST TREAT THE CSV AND PARSE IT TO DF
    csv_path = f"{path}/moteval.csv"
    df = pd.read_csv(csv_path, sep=",")
    df = df.apply(lambda row: transform_str_to_list(row, [
        'results_responses', 'results_rt', 'results_speed_stim', 'results_correct']), axis=1)
    df = delete_uncomplete_participants(df)
    df = df.apply(compute_mean_per_condition, axis=1)
    df.to_csv(f'{path}/moteval_treat.csv')
    nb_trials = len(df['results_correct'][0])
    conditions = [1, 4, 8]
    outcomes_names_acc = [f"{cdt}-accuracy" for cdt in conditions]
    outcomes_names_rt = [f"{cdt}-rt" for cdt in conditions]
    base = ['participant_id', 'task_status', 'condition']
    df = df[base + outcomes_names_acc + outcomes_names_rt]
    for cdt in conditions:
        df[f'{cdt}-nb'] = 15
        df[f'{cdt}-correct'] = df[f'{cdt}-accuracy'] * 15
    df['total-task-correct'] = convert_to_global_task(df, [f"{col}-correct" for col in conditions])
    df['total-task-accuracy'] = df[[f"{col}" for col in outcomes_names_acc]].mean(axis=1)
    df['total-task-nb'] = 45
    if save_lfa:
        df.to_csv(f'{path}/moteval_lfa.csv', index=False)
    return df


# ## RUN FITTED MODELS AND PLOT VISUALISATIONS ###
def retrieve_zpdes_vs_baseline(study, conditions_to_keep, model_type, model=None):
    task = "moteval"
    path = f"../outputs/{study}/results_{study}/{task}"
    df = format_data(path)
    condition_list = [1, 4, 8, 'total-task']
    root_path = f"{study}-{model_type}"
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
    task = "moteval"
    path = f"../outputs/{study}/results_{study}/{task}"
    df = format_data(path)
    if model:
        get_pymc_trace(df, conditions_to_fit, task=task, model_object=model, model_type=model_type, study=study)


def get_stan_accuracy(dataframe, outcomes_names_acc, nb_trials, study):
    # BAYES ACCURACY ANALYSIS
    # For accuracy analysis, let's focus on the outcomes:
    stan_distributions = get_stan_accuracy_distributions(dataframe, outcomes_names_acc, nb_trials, 'moteval')
    # Draw figures for accuracy data
    plt_args = {'list_xlim': [-0.25, 2.25], 'list_ylim': [0.4, 1],
                'list_set_xticklabels': ['1', '4', '8'], 'list_set_xticks': [0, 1, 2],
                'list_set_yticklabels': ['2', '3', '4', '5'],
                'list_set_yticks': [0.4, 0.6, 0.8, 1.0],
                'scale_jitter': 0.3}
    plot_all_accuracy_figures(stan_distributions=stan_distributions, outcomes_names=outcomes_names_acc,
                              task_name='moteval', overall_initial_data=dataframe, nb_trials=nb_trials,
                              plot_args=plt_args, study=study)


if __name__ == '__main__':
    study = "v0_axa"
    # fit_model(study)
    # -------------------------------------------------------------------#
    # BAYES RT ANALYSIS:
    # '''
    # stan_rt_distributions = get_stan_RT_distributions(dataframe, ["1", "4", "8"])
    # plt_args = {'list_xlim': [0.75, 3.25], 'list_ylim': [0, 1],
    #             'list_set_xticklabels': ['1', '4', '8'], 'list_set_xticks': [1, 2, 3],
    #             'list_set_yticklabels': ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
    #             'list_set_yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0]}
    # plot_all_rt_figures(stan_rt_distributions, outcomes_names_rt, dataframe=dataframe, task_name='moteval',
    #                     plot_args=plt_args)
    # '''
