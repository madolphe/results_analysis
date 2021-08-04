import copy
from sklearn.linear_model import LinearRegression
from utils import *
from cal_stan_accuracy_rt import CalStan_accuracy, CalStan_rt


# keyRes1 = F => 1 (ODD impair - LOW)
# keyRes2 = J => 2 (EVEN pair - HIGH)
# task1 = parity (0)
# task2 = relative (1)

def delete_uncomplete_participants(dataframe):
    """

    """
    mask = pd.DataFrame(dataframe.participant_id.value_counts() < 2)
    participants_to_delete = mask[mask['participant_id'] == True].index.tolist()
    for id in participants_to_delete:
        dataframe = dataframe[dataframe['participant_id'] != id]
    return dataframe


def delete_beggining_of_block(row):
    results = row["results_ind_switch"].split(",")
    results = [int(elt) for elt in results]
    new_row = copy.deepcopy(results)
    for idx, elt in enumerate(results):
        if idx % 33 == 0:
            new_row[idx] = 0
    return new_row


def transform_string_to_row(row, column):
    return [int(elt) for elt in row[column].split(',') if elt]


def correct_sequence_of_answers(row):
    seq_answer_relative = []
    seq_answer_parity = []
    seq_relative_switch = []
    seq_parity_switch = []
    seq_relative_rt = []
    seq_parity_rt = []
    for response, task, target, switch, rt, ind in zip(row.results_responses, row.results_indtask,
                                                       row.results_trial_target,
                                                       row.results_ind_switch_clean, row.results_rt,
                                                       range(len(row.results_ind_switch_clean))):

        if ind != 0 and ind != 33 and ind != 66:  # to exclude the first trials
            # First check what activity is requested - if None => do not consider the trial
            if task == 1:
                seq_relative_switch.append(switch)
                seq_relative_rt.append(rt)
                if (response == 1 and target < 5) or (response == 2 and target > 5):
                    seq_answer_relative.append(1)
                else:
                    seq_answer_relative.append(0)
            elif task == 0:
                seq_parity_switch.append(switch)
                seq_parity_rt.append(rt)
                if (response == 1 and (target % 2) == 1) or (response == 2 and (target % 2) == 0):
                    seq_answer_parity.append(1)
                else:
                    seq_answer_parity.append(0)
    return seq_answer_relative, seq_answer_parity, seq_relative_switch, seq_parity_switch, seq_relative_rt, seq_parity_rt


def compute_correct_answer(row, answer_type):
    seq_answer_relative, seq_answer_parity, seq_relative_switch, seq_parity_switch, seq_relative_rt, seq_parity_rt = correct_sequence_of_answers(
        row)
    if answer_type == "correct_total":
        return seq_answer_relative.count(1) + seq_answer_parity.count(1)
    elif answer_type == "correct_relative":
        return seq_answer_relative.count(1)
    elif answer_type == "correct_parity":
        return seq_answer_parity.count(1)
    elif answer_type == "total_nb":
        return len(seq_answer_parity) + len(seq_answer_relative)
    elif answer_type == "parity_nb":
        return len(seq_answer_parity)
    elif answer_type == "relative_nb":
        return len(seq_answer_relative)
    elif answer_type == "check_switch":
        parity_errors_switch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 1 and elt == 0)])
        relative_errors_switch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 1 and elt == 0)])
        return parity_errors_switch + relative_errors_switch
    # summarize the relative and parity condition for accuracy
    elif answer_type == "check_switch_hit":
        parity_hit_switch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 1 and elt == 1)])
        relative_hit_switch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 1 and elt == 1)])
        return parity_hit_switch + relative_hit_switch
    elif answer_type == "check_unswitch_hit":
        parity_hit_unswitch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 0 and elt == 1)])
        relative_hit_unswitch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 0 and elt == 1)])
        return parity_hit_unswitch + relative_hit_unswitch
    # separate the relative and parity condition for accuracy
    elif answer_type == "parity_check_switch_hit":
        parity_hit_switch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 1 and elt == 1)])
        return parity_hit_switch
    elif answer_type == "relative_check_switch_hit":
        relative_hit_switch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 1 and elt == 1)])
        return relative_hit_switch
    elif answer_type == "parity_check_unswitch_hit":
        parity_hit_unswitch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 0 and elt == 1)])
        return parity_hit_unswitch
    elif answer_type == "relative_check_unswitch_hit":
        relative_hit_unswitch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 0 and elt == 1)])
        return relative_hit_unswitch
    # total number for each conditions
    elif answer_type == "parity_check_switch_total":
        parity_total_switch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 1)])
        return parity_total_switch
    elif answer_type == "relative_check_switch_total":
        relative_total_switch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 1)])
        return relative_total_switch
    elif answer_type == "parity_check_unswitch_total":
        parity_total_unswitch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 0)])
        return parity_total_unswitch
    elif answer_type == "relative_check_unswitch_total":
        relative_hit_unswitch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 0)])
        return relative_hit_unswitch
    # summarize the relative and parity condition for rt
    elif answer_type == "check_switch_rt":
        parity_rt_switch = sum(
            [rt for elt, rt, sw in zip(seq_answer_parity, seq_parity_rt, seq_parity_switch) if (sw == 1)])
        relative_rt_switch = sum(
            [rt for elt, rt, sw in zip(seq_answer_relative, seq_relative_rt, seq_relative_switch) if (sw == 1)])
        return (parity_rt_switch + relative_rt_switch) / (len(seq_parity_rt) + len(seq_relative_rt))
    elif answer_type == "check_unswitch_rt":
        parity_rt_unswitch = sum(
            [rt for elt, rt, sw in zip(seq_answer_parity, seq_parity_rt, seq_parity_switch) if (sw == 0 and elt == 1)])
        relative_rt_unswitch = sum(
            [rt for elt, rt, sw in zip(seq_answer_relative, seq_relative_rt, seq_relative_switch) if (sw == 0)])
        return (parity_rt_unswitch + relative_rt_unswitch) / (len(seq_parity_rt) + len(seq_relative_rt))
    # separate the relative and parity condition for rt
    elif answer_type == "parity_check_switch_hit":
        parity_hit_switch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 1 and elt == 1)])
        return parity_hit_switch
    elif answer_type == "relative_check_switch_hit":
        relative_hit_switch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 1 and elt == 1)])
        return relative_hit_switch
    elif answer_type == "parity_check_unswitch_hit":
        parity_hit_unswitch = sum(
            [1 for elt, sw in zip(seq_answer_parity, seq_parity_switch) if (sw == 0 and elt == 1)])
        return parity_hit_unswitch
    elif answer_type == "relative_check_unswitch_hit":
        relative_hit_unswitch = sum(
            [1 for elt, sw in zip(seq_answer_relative, seq_relative_switch) if (sw == 0 and elt == 1)])
        return relative_hit_unswitch
    elif answer_type == "parity_check_switch_rt":
        parity_rt_switch = sum(
            [rt for elt, rt, sw in zip(seq_answer_parity, seq_parity_rt, seq_parity_switch) if (sw == 1)])
        return (parity_rt_switch) / len(seq_parity_rt)
    elif answer_type == "relative_check_switch_rt":
        relative_rt_switch = sum(
            [rt for elt, rt, sw in zip(seq_answer_relative, seq_relative_rt, seq_relative_switch) if (sw == 1)])
        return (relative_rt_switch) / len(seq_relative_rt)
    elif answer_type == "parity_check_unswitch_rt":
        parity_rt_unswitch = sum(
            [rt for elt, rt, sw in zip(seq_answer_parity, seq_parity_rt, seq_parity_switch) if (sw == 0 and elt == 1)])
        return (parity_rt_unswitch) / len(seq_parity_rt)
    elif answer_type == "relative_check_unswitch_rt":
        relative_rt_unswitch = sum(
            [rt for elt, rt, sw in zip(seq_answer_relative, seq_relative_rt, seq_relative_switch) if (sw == 0)])
        return (relative_rt_unswitch) / len(seq_relative_rt)


def compute_mean(row):
    return np.mean(row["results_rt"])


def boxplot_pre_post(column, figname):
    pre_test = dataframe[dataframe['task_status'] == 'PRE_TEST'][column]
    post_test = dataframe[dataframe['task_status'] == 'POST_TEST'][column]
    plt.boxplot([pre_test.values, post_test.values], positions=[0, 1])
    plt.xticks([0, 1], ['PRE-TEST', 'POST-TEST'])
    plt.savefig(f"../outputs/taskswitch/{figname}.png")
    plt.close()


def linear_reg_and_plot(column, figname):
    post_test = dataframe[dataframe['task_status'] == 'POST_TEST'][column]
    pre_test = dataframe[dataframe['task_status'] == 'PRE_TEST'][column]
    reg = LinearRegression().fit(np.expand_dims(pre_test.values, axis=1), post_test.values)
    score = reg.score(np.expand_dims(pre_test.values, axis=1), post_test.values)
    plt.scatter(x=pre_test, y=post_test, c='red')
    plt.plot([pre_test.values.min(), pre_test.values.max()],
             reg.predict(np.expand_dims([pre_test.values.min(), pre_test.values.max()], axis=1)), color='blue',
             linewidth=3)
    plt.title(f"R**2 : {score}")
    plt.savefig(f"../outputs/taskswitch/{figname}.png")
    plt.close()


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


def get_overall_dataframe_taskswitch(dataframe, outcomes_names):
    # summarize two days experiments
    indices_id = extract_id(dataframe, num_count=2)
    sum_observers, tmp_nb = [], []
    for ob in indices_id:
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        sum_observers.append([np.sum(tmp_df[index]) for index in outcomes_names])
    sum_observers = pd.DataFrame(sum_observers, columns=outcomes_names)
    return sum_observers


if __name__ == '__main__':
    # -------------------------------------------------------------------#
    # DATAFRAME CREATION
    csv_path = "../outputs/taskswitch/taskswitch.csv"
    dataframe = pd.read_csv(csv_path)
    dataframe = delete_uncomplete_participants(dataframe)
    dataframe["results_responses"] = dataframe.apply(lambda row: transform_string_to_row(row, "results_responses"),
                                                     axis=1)
    dataframe["results_trial_target"] = dataframe.apply(
        lambda row: transform_string_to_row(row, "results_trial_target"), axis=1)
    dataframe["results_indtask"] = dataframe.apply(
        lambda row: transform_string_to_row(row, "results_indtask"), axis=1)
    dataframe["results_rt"] = dataframe.apply(
        lambda row: transform_string_to_row(row, "results_rt"), axis=1)
    print(dataframe.info())
    # results_ind_switch : remove first element of each row by null
    # 3 blocks - 99 responses (idx: 0 - 33 - 66 , beggining of each block should be set to null)
    # participant = dataframe[dataframe['task_status'] == "PRE_TEST"]
    # participant = participant[participant['participant_id'] == 15]
    dataframe["results_ind_switch_clean"] = dataframe.apply(delete_beggining_of_block, axis=1)

    # results_response: actual answer of the participant
    # ind_switch: is it a "reconfiguration answer" 1=lower-even / 2=higher-odd
    # results_trial_target: is the question
    dataframe["nb_correct_total_answer"] = dataframe.apply(lambda row: compute_correct_answer(row, "correct_total"),
                                                           axis=1)
    dataframe["nb_correct_relative_answer"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "correct_relative"), axis=1)
    dataframe["nb_correct_parity_answer"] = dataframe.apply(lambda row: compute_correct_answer(row, "correct_parity"),
                                                            axis=1)
    dataframe["nb_total"] = dataframe.apply(lambda row: compute_correct_answer(row, "total_nb"), axis=1)
    dataframe["nb_parity"] = dataframe.apply(lambda row: compute_correct_answer(row, "parity_nb"), axis=1)
    dataframe["nb_relative"] = dataframe.apply(lambda row: compute_correct_answer(row, "relative_nb"), axis=1)
    dataframe["errors_in_switch"] = dataframe.apply(lambda row: compute_correct_answer(row, "check_switch"), axis=1)
    dataframe["total_error"] = dataframe["nb_total"] - dataframe["nb_correct_total_answer"]
    dataframe["accuracy"] = dataframe["nb_correct_total_answer"] / dataframe["nb_total"]

    # Plot accuracy
    boxplot_pre_post("accuracy", "accuracy")

    # Mean RT
    dataframe["mean_RT"] = dataframe.apply(compute_mean, axis=1)
    boxplot_pre_post("mean_RT", "reaction_time_mean")
    linear_reg_and_plot("mean_RT", "linear_reg_RT")
    plt.close()

    dataframe.to_csv("../outputs/taskswitch/taskswitch_treatment.csv")

    # -------------------------------------------------------------------#
    # LATENT FACTOR ANALYSIS DF CREATION
    # from here written by mswym
    # Additional condition extration:
    dataframe["correct_in_switch"] = dataframe.apply(lambda row: compute_correct_answer(row, "check_switch_hit"),
                                                     axis=1)
    dataframe["correct_in_unswitch"] = dataframe.apply(lambda row: compute_correct_answer(row, "check_unswitch_hit"),
                                                       axis=1)
    dataframe["switch-rt"] = dataframe.apply(lambda row: compute_correct_answer(row, "check_switch_rt"), axis=1)
    dataframe["unswitch-rt"] = dataframe.apply(lambda row: compute_correct_answer(row, "check_unswitch_rt"), axis=1)

    # (relative or parity) AND (switch OR unswitch) = 4 cases
    # 2 outcomes taken into account : nb correct and rt
    dataframe["relative-switch-correct"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "relative_check_switch_hit"), axis=1)
    dataframe["relative-unswitch-correct"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "relative_check_unswitch_hit"), axis=1)
    dataframe["relative-switch-nb"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "relative_check_switch_total"), axis=1)
    dataframe["relative-unswitch-nb"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "relative_check_unswitch_total"), axis=1)
    dataframe["relative-switch-rt"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "relative_check_switch_rt"), axis=1)
    dataframe["relative-unswitch-rt"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "relative_check_unswitch_rt"), axis=1)

    dataframe["parity-switch-correct"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "parity_check_switch_hit"), axis=1)
    dataframe["parity-unswitch-correct"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "parity_check_unswitch_hit"), axis=1)
    dataframe["parity-switch-nb"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "parity_check_switch_total"), axis=1)
    dataframe["parity-unswitch-nb"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "parity_check_unswitch_total"), axis=1)
    dataframe["parity-switch-rt"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "parity_check_switch_rt"), axis=1)
    dataframe["parity-unswitch-rt"] = dataframe.apply(
        lambda row: compute_correct_answer(row, "parity_check_unswitch_rt"), axis=1)

    # sumirize two days experiments
    sum_observers = []
    sum_observers_forsave = []
    condition_names = ['parity-switch', 'parity-unswitch', 'relative-switch', 'relative-unswitch']
    condition_names_accuracy = [f"{condition}-accuracy" for condition in condition_names]
    condition_names_rt = [f"{condition}-rt" for condition in condition_names]

    # extract observer index information
    indices_id = extract_id(dataframe, num_count=2)
    for ob in indices_id:
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        participant_tmp = []
        participant_tmp.append(ob)
        for condition in condition_names:
            participant_tmp.append(np.sum(tmp_df[f"{condition}-correct"]))
            participant_tmp.append(np.sum(tmp_df[f"{condition}-nb"]))
            participant_tmp.append(np.mean(tmp_df[f"{condition}-correct"] / tmp_df[f"{condition}-nb"]))
            participant_tmp.append(np.mean(tmp_df[f"{condition}-rt"]))
        sum_observers.append(participant_tmp)
    columns, keywords = ['participant_id'], ['correct', 'nb', 'accuracy', 'rt']
    for condition in condition_names:
        for keyword in keywords:
            columns.append(f"{condition}-{keyword}")
    sum_observers = pd.DataFrame(sum_observers, columns=columns)
    # for save summary data
    sum_observers.to_csv('../outputs/taskswitch/sumdata_taskswitch.csv', header=True, index=False)
    # -------------------------------------------------------------------#
    # BAYES ACCURACY :
    nb_trials_names = [f"{condition}-nb" for condition in condition_names]
    condition_names_correct = [f"{condition}-correct" for condition in condition_names]

    # # Task number is not always the same:
    pretest, posttest = get_pre_post_dataframe(dataframe, condition_names_correct + nb_trials_names)
    # # Get mean data for
    sum_observers = get_overall_dataframe_taskswitch(dataframe, condition_names_correct + nb_trials_names)
    # # Compute stan_accuracy for all conditions:
    stan_sessions = [[], [], []]
    sessions = [sum_observers, pretest, posttest]
    for condition, condition_nb in zip(condition_names_correct, nb_trials_names):
        for idx, session in enumerate(sessions):
            session['total_resp'] = session[condition_nb]
            stan_sessions[idx].append(CalStan_accuracy(session, ind_corr_resp=condition))
    class_stan_accuracy_overall, class_stan_accuracy_pretest, class_stan_accuracy_posttest = stan_sessions
    # Group all stan distributions into a dict:
    stan_distributions = {'overall': class_stan_accuracy_overall,
                          'pretest': class_stan_accuracy_pretest,
                          'posttest': class_stan_accuracy_posttest}
    # Draw figures for accuracy data
    plot_args = {
        'list_xlim': [0.75, 4.25], 'list_ylim': [0, 1],
        'list_set_xticklabels': ['sw/oe', 'no-sw/oe', 'sw/hl', 'no-sw/hl'],
        'list_set_xticks': [1, 2, 3, 4],
        'list_set_yticklabels': ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
        'list_set_yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    }
    plot_prepost_mean_accuracy_distribution(condition_names_correct, stan_distributions,
                                            f'../outputs/taskswitch/taskswitch_distrib_reliability.png')

    # -------------------------------------------------------------------#
    # BAYES RT ANALYSIS:
    stan_rt_distributions = get_stan_RT_distributions(dataframe, condition_names)
    plt_args = {"list_xlim": [0.5, 2.5], "list_ylim": [-100, 250],
                "list_set_xticklabels": ['odd-even', 'high-low'], "list_set_xticks": [1, 2],
                "list_set_yticklabels": ['-100', '0', '100', '200'], "list_set_yticks": [-100, 0, 100, 200],
                "val_ticks": 10}
    plot_all_rt_figures(stan_rt_distributions, condition_names_rt, dataframe=dataframe, task_name='taskswitch',
                        plot_args=plt_args)
    print('finished')
