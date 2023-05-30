# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5422529/
import copy

from sklearn.linear_model import LinearRegression
from utils import *
from pymc.data import change_accuracy_for_correct_column, convert_to_global_task
from utils import retrieve_and_init_models, add_difference_pre_post, get_pymc_trace
from pathlib import Path

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


def find_participant_with_fewer_blocks(dataframe: pd.DataFrame) -> (str, int, int):
    """
    get participant with lowest nb of blocks
    find how many GO (nb of 3) and no GO blocks (len - nb of 3) have been recorded
    """
    row = dataframe.loc[dataframe['nb_blocks'].idxmin()]
    participant_id, nb_blocks = row['participant_id'], row['nb_blocks']
    nb_go = row['results_targetvalue'].count("3")
    return participant_id, nb_blocks, nb_go


def delete_non_recorded_blocks(row, nb_blocks):
    count_go = 0
    count_no_go = 0
    idx = 0
    dict_columns = {'results_responses': [], 'results_rt': [], 'results_ind_previous': [], 'results_targetvalue': []}

    assert len(row['results_responses']) == len(row['results_rt']) == len(row['results_ind_previous']) == len(
        row['results_targetvalue'])

    while idx < len(row['results_targetvalue']):
        if row['results_targetvalue'][idx] == '3':
            count_go += 1
            if count_go <= nb_blocks:
                for column_name, column_list in dict_columns.items():
                    column_list.append(row[column_name][idx])
        else:
            count_no_go += 1
            if count_no_go <= nb_blocks:
                for column_name, column_list in dict_columns.items():
                    column_list.append(row[column_name][idx])
        idx += 1

    assert len(dict_columns['results_responses']) == len(dict_columns['results_rt']) == len(
        dict_columns['results_ind_previous']) == len(dict_columns['results_targetvalue'])

    assert len(dict_columns['results_responses']) == (2 * nb_blocks)

    for column_name, column_list in dict_columns.items():
        row[column_name] = column_list
    return row


def parse_to_int(elt: str) -> int:
    """
        Parse string value into int, if string is null parse to 0
        If null; participant has not pressed the key when expected
    """
    if elt == '':
        return 0
    return int(elt)


def compute_nb_commission_errors(row: pd.core.series.Series) -> int:
    """
        Take a dataframe row and returns the number of element 2 in the result response
        2 == commission error
    """
    results_responses = list(
        map(parse_to_int, row["results_responses"]))  # 1 is click when expected, 0 no click when expected, 2 is mistake
    return results_responses.count(2)


def compute_number_of_keyboard_input(row: pd.core.series.Series) -> int:
    return len(row["results_responses"])


def transform_2_in_0(elt: int) -> int:
    """
        Take an int and returns the remainder in the euclidean division
        ("0 - 1 - 2" possible values and we want to transform 2 into 0)
    """
    return elt % 2


def list_of_correct_hits(row: pd.core.series.Series) -> list:
    """
        Take a dataframe row and returns a list of rt for hits (not for commission errors)
    """
    results_rt = list(map(parse_to_int, row["results_rt"]))  # RT when interraction could happen (after 7)
    results_responses = list(map(parse_to_int, row["results_responses"]))
    mask = list(map(transform_2_in_0, results_responses))
    rt_hits = [a * b for a, b in zip(results_rt, mask)]
    return rt_hits


def compute_means(row: pd.core.series.Series) -> float:
    """
        Useless function that returns the mean on a row (pandas already provides one)
    """
    tmp = np.array(row['result_clean_rt'])
    # this was changed to calculate only the hit trials by mswym
    # print(np.mean(tmp[np.where(tmp!=0)]))
    # print(np.mean(row['result_clean_rt']))
    return np.mean(tmp[np.where(tmp != 0)])


def compute_number_of_omissions(row: pd.core.series.Series) -> int:
    """
        From the row of results, return the number of 0 if elt is 3 in results_targetvalue
    """
    results_responses = list(map(parse_to_int, row["results_responses"]))
    results_targetvalue = list(map(parse_to_int, row["results_targetvalue"]))
    count = 0
    for idx, elt in enumerate(results_targetvalue):
        if elt == 3:
            if results_responses[idx] == 0:
                count += 1
    return count


def compute_result_sum_hr(row):
    return 18 - row['NOGO-correct']


def compute_numbercond(row, ind_cond):
    """
        From the row of results, return the list of resuluts_correct according to results_num_stim condition 
    """
    results_responses = list(row["result_correct"])
    results_targetvalue = [int(t) for t in row["results_targetvalue"].split(',')]
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


def extract_mu_ci_from_summary_rt(dataframe):
    out = np.zeros((1, 3))  # 3 means the mu, ci_min, and ci_max
    out[0, 0] = dataframe.mu_rt
    out[0, 1] = dataframe.ci_min
    out[0, 2] = dataframe.ci_max
    return out


def plt_commision_errors(dataframe):
    # commission errors (i.e., falsely pressing the button in no-go trials; also called false alarms)
    dataframe.groupby(["task_status", "result_commission_errors"]).count()["participant_id"].unstack(
        "task_status")[['PRE_TEST', 'POST_TEST']].plot.bar()
    plt.title("Participants in number of commission errors c")
    plt.savefig("../outputs/gonogo/gonogo_commissions.png")
    plt.close()


def plt_reliability(pre_test, post_test):
    reg = LinearRegression().fit(np.expand_dims(pre_test.values, axis=1), post_test.values)
    score = reg.score(np.expand_dims(pre_test.values, axis=1), post_test.values)
    plt.scatter(x=pre_test, y=post_test, c='red')
    plt.plot([pre_test.values.min(), pre_test.values.max()],
             reg.predict(np.expand_dims([pre_test.values.min(), pre_test.values.max()], axis=1)), color='blue',
             linewidth=3)
    plt.title(f"R**2 : {score}")
    plt.savefig("../outputs/gonogo/rt_pre_post_gonogo.png")
    plt.close()


def plt_omission_errors(dataframe):
    # omission errors (i.e., falsely not pressing the button in go trials; also called misses):
    dataframe.groupby(["task_status", "result_nb_omission"]).count()["participant_id"].unstack("task_status")[
        ['PRE_TEST', 'POST_TEST']].plot.bar()
    plt.title("Number of omission errors per participant")
    plt.savefig("../outputs/gonogo/gonogo_ommissions.png")
    plt.close()


def format_data(path, save_lfa):
    df = pd.read_csv(f"{path}/gonogo.csv", sep=",")
    df = df.apply(lambda row: transform_str_to_list(row, [
        'results_responses', 'results_rt', 'results_ind_previous', 'results_targetvalue']), axis=1)
    df['NOGO-correct'] = df.apply(compute_number_of_omissions, axis=1)
    df['result_commission_errors'] = df.apply(compute_nb_commission_errors, axis=1)
    # dataframe = delete_uncomplete_participants(df)
    dataframe = df
    # false alarm relative to sequence length
    df['nb_blocks'] = df.apply(compute_number_of_keyboard_input, axis=1)
    participant_id, nb_blocks, nb_go = find_participant_with_fewer_blocks(dataframe)
    blocks_list = [nb_go, nb_blocks - nb_go]
    NB_BLOCKS_TO_KEEP = min(blocks_list)
    is_go_blocks = blocks_list.index(NB_BLOCKS_TO_KEEP) == 0
    print(f"ID {participant_id} has the smallest nb of blocks recorded ({nb_blocks}) with {nb_go} go blocks.")
    print(f"Nb of blocks to keep: {NB_BLOCKS_TO_KEEP}")
    print(f"Blocks to keep are go blocks: {is_go_blocks}")
    df = df.apply(lambda row: delete_non_recorded_blocks(row, NB_BLOCKS_TO_KEEP), axis=1)
    df['nb_blocks'] = df.apply(compute_number_of_keyboard_input, axis=1)
    # Reaction times in or the number of correct go-trials (i.e., hits):
    df['result_clean_rt'] = df.apply(list_of_correct_hits, axis=1)
    df['HR-nb'] = df.apply(lambda row: len(row['result_clean_rt']), axis=1)
    df['GO-rt'] = df.apply(compute_means, axis=1)
    df['GO-correct'] = df.apply(compute_result_sum_hr, axis=1)
    df['GO-accuracy'] = df.apply(lambda row: row['GO-correct'] / NB_BLOCKS_TO_KEEP, axis=1)
    df['NOGO-accuracy'] = df.apply(lambda row: row['NOGO-correct'] / NB_BLOCKS_TO_KEEP, axis=1)
    conditions = ['GO', 'NOGO', 'total-task']
    conditions_accuracy = [f'{cdt}-accuracy' for cdt in conditions]
    conditions_correct = [f'{cdt}-correct' for cdt in conditions]
    conditions_nb = [f'{cdt}-nb' for cdt in conditions]
    for cdt in conditions_nb:
        df[cdt] = NB_BLOCKS_TO_KEEP
    conditions_RT = ['GO-rt']
    base = ['participant_id', 'task_status', 'condition']
    df['total-task-correct'] = df['GO-correct'] + (NB_BLOCKS_TO_KEEP - df['NOGO-correct'])
    df['total-task-nb'] = NB_BLOCKS_TO_KEEP * 2
    df['total-task-accuracy'] = df['total-task-correct'] / df['total-task-nb']
    df = df[base + conditions_accuracy + conditions_RT + conditions_nb + conditions_correct]
    if save_lfa:
        df.to_csv(f"{path}/gonogo_lfa.csv", index=False)
    return df


# ## RUN FITTED MODELS AND PLOT VISUALISATIONS ###
def retrieve_zpdes_vs_baseline(study, conditions_to_keep, model_type, model=None):
    task = "gonogo"
    path = f"../outputs/{study}/results_{study}/{task}"
    root_path = f"{study}-{model_type}"
    df = format_data(path, save_lfa=False)
    condition_list = ['GO', 'NOGO', 'total-task']
    columns = [f"{c}-accuracy" for c in condition_list] + [f"{c}-correct" for c in condition_list]
    df = pd.concat([df, add_difference_pre_post(df, columns)], axis=0)
    model_baseline = retrieve_and_init_models(root_path, task, conditions_to_keep, df, model, group="baseline")
    model_zpdes = retrieve_and_init_models(root_path, task, conditions_to_keep, df, model, group="zpdes")
    return model_zpdes, model_baseline


def run_visualisation(study, conditions_to_keep, model_type, model=None):
    model_zpdes, model_baseline = retrieve_zpdes_vs_baseline(study, conditions_to_keep, model_type, model)
    model_baseline.plot_posterior_and_population()
    model_baseline.plot_comparison_posterior_and_population(model_zpdes)


# ## RUN FITTED MODELS AND PLOT VISUALISATIONS ###
def fit_model(study, conditions_to_fit, model=None, model_type="pooled_model", save_lfa=False):
    task = "gonogo"
    savedir = f"../outputs/{study}/results_{study}"
    path = f"{savedir}/{task}"
    Path(savedir).mkdir(parents=True, exist_ok=True)
    df = format_data(path, save_lfa=save_lfa)
    if model:
        get_pymc_trace(df, conditions_to_fit, task=task, model_object=model, model_type=model_type, study=study)


def get_stan_RT(dataframe, conditions_full_names, study):
    values_conditions = ["HR"]
    stan_rt_distributions = get_stan_RT_distributions(dataframe, values_conditions, 'gonogo')
    plt_args = {'list_xlim': [-0.5, 0.5], 'list_ylim': [0, 1],
                'list_set_xticklabels': ['HR-rt'], 'list_set_xticks': [0],
                'list_set_yticklabels': ['0', '250', '500', '750'], 'list_set_yticks': [0, 250, 500, 750],
                'val_ticks': 25,
                'scale_jitter': 0.1,
                'scale_panel': 2}
    plot_all_rt_figures(stan_rt_distributions, conditions_full_names, dataframe=dataframe, task_name='gonogo',
                        plot_args=plt_args, study=study)


def get_stan_accuracy(dataframe, outcomes_names, nb_trials, study):
    stan_distributions = get_stan_accuracy_distributions(dataframe, outcomes_names, nb_trials, 'gonogo')
    # Draw figures for accuracy data

    plot_args = {'list_xlim': [-0.5, 1.5], 'list_ylim': [0, 1],
                 'list_set_xticklabels': ['HR', 'FAR'], 'list_set_xticks': [0, 1],
                 'list_set_yticklabels': ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                 'list_set_yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                 'scale_jitter': 0.2}
    plot_all_accuracy_figures(stan_distributions=stan_distributions, outcomes_names=outcomes_names,
                              task_name='gonogo', overall_initial_data=dataframe, nb_trials=nb_trials,
                              plot_args=plot_args, study=study)


if __name__ == '__main__':
    study = "v0_axa"
    # fit_model(study)
    # -------------------------------------------------------------------#
    # Bayes accuracy analysis:
    # get_stan_accuracy(dataframe, outcomes_names, nb_trials, study)
    # -------------------------------------------------------------------#
    # BAYES RT ANALYSIS:
    # get_stan_RT(dataframe, conditions_full_names, study)
    # print('finished')
