# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5422529/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from cal_stan_accuracy_rt import CalStan_accuracy, CalStan_rt


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
    return 18 - row['result_nb_omission']


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
    axes.set_xticks([1, 2])
    axes.set_xlim([0.5, 2.5])
    axes.set_xticklabels(['HR', 'FAR'], fontsize=20)
    axes.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=20)
    fig.savefig(fname_save)
    plt.show()


def draw_all_distributions_rt(dist_ind, dist_summary, num_dist, num_cond, std_val=0.05,
                              fname_save='memorability_rt.png'):
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
    axes.set_xlim([0.5, 1.5])
    axes.set_xticks([1])
    axes.set_xticklabels(['HR'], fontsize=20)
    axes.set_yticks([0, 250, 500, 750])
    axes.set_yticklabels(['0', '250', '500', '750'], fontsize=20)
    fig.savefig(fname_save)
    plt.show()


def extract_mu_ci_from_summary_rt(dataframe):
    out = np.zeros((1, 3))  # 3 means the mu, ci_min, and ci_max
    out[0, 0] = dataframe.mu_rt
    out[0, 1] = dataframe.ci_min
    out[0, 2] = dataframe.ci_max
    return out


if __name__ == '__main__':
    csv_path = "../outputs/gonogo/gonogo.csv"
    dataframe = pd.read_csv(csv_path)

    dataframe = dataframe.apply(lambda row: transform_str_to_list(row, [
        'results_responses', 'results_rt', 'results_ind_previous', 'results_targetvalue']), axis=1)

    dataframe = delete_uncomplete_participants(dataframe)

    # false alarm relative to sequence length
    dataframe['nb_blocks'] = dataframe.apply(compute_number_of_keyboard_input, axis=1)

    participant_id, nb_blocks, nb_go = find_participant_with_fewer_blocks(dataframe)
    blocks_list = [nb_go, nb_blocks - nb_go]
    NB_BLOCKS_TO_KEEP = min(blocks_list)
    is_go_blocks = blocks_list.index(NB_BLOCKS_TO_KEEP) == 0
    print(f"ID {participant_id} has the smallest nb of blocks recorded ({nb_blocks}) with {nb_go} go blocks.")
    print(f"Nb of blocks to keep: {NB_BLOCKS_TO_KEEP}")
    print(f"Blocks to keep are go blocks: {is_go_blocks}")
    dataframe = dataframe.apply(lambda row: delete_non_recorded_blocks(row, NB_BLOCKS_TO_KEEP), axis=1)
    dataframe['nb_blocks'] = dataframe.apply(compute_number_of_keyboard_input, axis=1)

    # commission errors (i.e., falsely pressing the button in no-go trials; also called false alarms)
    dataframe['result_commission_errors'] = dataframe.apply(compute_nb_commission_errors, axis=1)
    dataframe.groupby(["task_status", "result_commission_errors"]).count()["participant_id"].unstack(
        "task_status")[['PRE_TEST', 'POST_TEST']].plot.bar()
    plt.title("Participants in number of commission errors c")
    plt.savefig("../outputs/gonogo/gonogo_commissions.png")
    plt.close()

    # Reaction times in or the number of correct go-trials (i.e., hits):
    dataframe['result_clean_rt'] = dataframe.apply(list_of_correct_hits, axis=1)
    dataframe['mean_result_clean_rt'] = dataframe.apply(compute_means, axis=1)
    post_test = dataframe[dataframe['task_status'] == 'POST_TEST']['mean_result_clean_rt']
    pre_test = dataframe[dataframe['task_status'] == 'PRE_TEST']['mean_result_clean_rt']
    reg = LinearRegression().fit(np.expand_dims(pre_test.values, axis=1), post_test.values)
    score = reg.score(np.expand_dims(pre_test.values, axis=1), post_test.values)
    plt.scatter(x=pre_test, y=post_test, c='red')
    plt.plot([pre_test.values.min(), pre_test.values.max()],
             reg.predict(np.expand_dims([pre_test.values.min(), pre_test.values.max()], axis=1)), color='blue',
             linewidth=3)
    plt.title(f"R**2 : {score}")
    plt.savefig("../outputs/gonogo/rt_pre_post_gonogo.png")
    plt.close()

    # omission errors (i.e., falsely not pressing the button in go trials; also called misses):
    dataframe['result_nb_omission'] = dataframe.apply(compute_number_of_omissions, axis=1)
    dataframe.groupby(["task_status", "result_nb_omission"]).count()["participant_id"].unstack("task_status")[
        ['PRE_TEST', 'POST_TEST']].plot.bar()
    plt.title("Number of omission errors per participant")
    plt.savefig("../outputs/gonogo/gonogo_ommissions.png")
    plt.close()

    # Save data
    dataframe.to_csv("../outputs/gonogo/gonogo_treatment.csv")

    # from here written by mswym
    # condition extraction
    dataframe['result_correct'] = dataframe.apply(compute_result_sum_hr, axis=1)
    # dataframe['result_nb_omission']

    # extract observer index information
    indices_id = extract_id(dataframe, num_count=2)

    # sumirize two days experiments
    sum_observers = []
    for ob in indices_id:
        print(ob)
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        sum_observers.append(
            [np.sum(tmp_df.result_correct), np.sum(tmp_df.result_nb_omission), np.mean(tmp_df.mean_result_clean_rt)])

    sum_observers = pd.DataFrame(sum_observers)
    # for save summary data
    tmp = sum_observers / 36.
    tmp.loc[:, 2] = tmp.loc[:, 2] * 36
    tmp.to_csv('../outputs/gonogo/sumdata_egonogo.csv', header=False, index=False)
    sum_observers['total_resp'] = sum_observers.apply(lambda row: 36, axis=1)  # two days task

    # calculate the mean distribution and the credible interval
    class_stan_accuracy = [CalStan_accuracy(sum_observers, ind_corr_resp=n) for n in range(2)]
    class_stan_rt = CalStan_rt(sum_observers, ind_rt=2, max_rt=1000)

    # draw figures
    # for accuracy data
    dist_ind = sum_observers.iloc[0:len(sum_observers), 0:2].values / 36.
    dist_summary = extract_mu_ci_from_summary_accuracy(class_stan_accuracy, [0, 1])
    draw_all_distributions(dist_ind, dist_summary, len(sum_observers), num_cond=2, std_val=0.05,
                           fname_save='../outputs/gonogo/gonogo_hrfar.png')

    dist_ind = sum_observers.iloc[0:len(sum_observers), 2].values
    dist_summary = extract_mu_ci_from_summary_rt(class_stan_rt)
    draw_all_distributions_rt(dist_ind, dist_summary, len(sum_observers), num_cond=1, std_val=0.05,
                              fname_save='../outputs/gonogo/gonogo_rt.png')
    print('finished')
