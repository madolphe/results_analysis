import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cal_stan_accuracy_rt import CalStan_accuracy, CalStan_rt
from matplotlib.ticker import MultipleLocator
import copy


def get_pymc_trace(data, condition_list, model_object, model_type, study, task, sample_size=10000):
    model_baseline = model_object(data[data['condition'] == 'baseline'],
                                  name=task, group='baseline', folder=f'{study}-{model_type}',
                                  stim_cond_list=condition_list,
                                  sample_size=sample_size)
    model_baseline.run()
    model_zpdes = model_object(data[data['condition'] == 'zpdes'],
                               name=task, group='zpdes', folder=f'{study}-{model_type}',
                               stim_cond_list=condition_list,
                               sample_size=sample_size)
    model_zpdes.run()


def add_difference_pre_post(df, conditions):
    new_df = df.groupby('participant_id')[conditions].diff().dropna()
    new_df['task_status'], new_df['participant_id'] = 'diff', df['participant_id'].unique()
    new_df['condition'] = df.groupby(['participant_id', 'condition']).median().reset_index()['condition'].values
    return new_df


def retrieve_and_init_models(root_path, task, condition_list, data, model, group, sample_size=4000):
    dict_traces = {
        condition: f"../outputs/{root_path}/{task}/{task}_{group}_results/traces/{task}_{group}-{condition}-trace" for
        condition in condition_list}
    model = model(data[data['condition'] == group],
                  name=task, group=group, folder=root_path,
                  stim_cond_list=condition_list, traces_path=dict_traces,
                  sample_size=4000)
    return model


def extract_mu_ci_from_summary_accuracy(dataframe, ind_cond):
    # 3 means the mu, ci_min, and ci_max
    out = np.zeros((len(ind_cond), 3))
    for t, ind in enumerate(ind_cond):
        out[t, 0] = dataframe[ind].mu_theta
        out[t, 1] = dataframe[ind].ci_min
        out[t, 2] = dataframe[ind].ci_max
    return out


def extract_id(dataframe, num_count):
    mask = pd.DataFrame(dataframe.participant_id.value_counts() == num_count)
    indices_id = mask[mask['participant_id'] == True].index.tolist()
    return indices_id


def extract_mu_ci_from_summary_rt(dataframe, ind_cond):
    outs = np.zeros((len(ind_cond), 3))  # 3 means the mu, ci_min, and ci_max
    for index, _ in enumerate(ind_cond):
        outs[index, 0] = dataframe[index].mu_rt
        outs[index, 1] = dataframe[index].ci_min
        outs[index, 2] = dataframe[index].ci_max
    return outs


def get_stan_RT_distributions(dataframe, conditons_number, name_task):
    """

    """
    # Get pre-test post-test
    conditions_rt = [f"{condition}-rt" for condition in conditons_number]
    conditions_nb = [f"{condition}-nb" for condition in conditons_number]
    pretest, posttest = get_pre_post_dataframe(dataframe, conditions_rt + conditions_nb)

    # Get accross sessions dataframe:
    sum_observers = get_overall_dataframe_rt(dataframe, conditions_rt + conditions_nb)
    sum_observers['total_resp'], pretest['total_resp'], posttest['total_resp'] = (None for i in range(3))
    observations = [sum_observers, pretest, posttest]

    # not tuned to rt data.
    # sum_observers = sum_observers.astype('int')
    # sum_observers.to_csv('../outputs/'+name_task+'/sumdata_'+name_task+'_rt.csv', header=True, index=False)

    # Prepare variables to handle results:
    # class_stan_rt_overall, class_stan_rt_pretest, class_stan_rt_posttest = [], [], []
    distributions_results = {'overall': [], 'pretest': [], 'posttest': []}

    for distrib_key, observation in zip(distributions_results.keys(), observations):
        for condition_rt, condition_nb in zip(conditions_rt, conditions_nb):
            observation['total_resp'] = observation[condition_nb]
            distributions_results[distrib_key].append(CalStan_rt(observation, ind_rt=condition_rt, max_rt=1000))
    return distributions_results


def plot_all_rt_figures(stan_distributions, condition_names, dataframe, task_name, plot_args, study, name_option=''):
    # 1) plot comparison
    plot_prepost_stan_distribution(condition_names, stan_distributions,
                                   f'../outputs/{study}/results_{study}/{task_name}/{task_name}{name_option}_distrib_reliability_rt.png',
                                   variable='mu')
    # 2) plot rt measure for all task
    sum_observers = get_overall_dataframe_rt(dataframe, condition_names)
    dist_ind = sum_observers[condition_names].values
    # stan_distributions[0] corresponds to mean between pre and post session
    dist_summary = extract_mu_ci_from_summary_rt(stan_distributions['overall'], condition_names)
    draw_all_distributions(dist_ind, dist_summary, len(sum_observers), num_cond=len(condition_names), std_val=0.05,
                           **plot_args,
                           fname_save=f'../outputs/{study}/results_{study}/{task_name}/{task_name}{name_option}_rt.png')


def plot_all_accuracy_figures(stan_distributions, outcomes_names, task_name, overall_initial_data, nb_trials,
                              plot_args, study, name_option=''):
    """
    This function plot histogram + kde for pre-post and mean between pre/post for the estimated parameter
    """
    plot_prepost_mean_accuracy_distribution(outcomes_names, stan_distributions,
                                            f'../outputs/{study}/results_{study}/{task_name}/{task_name}{name_option}_distrib_reliability_accuracy.png')
    dist_ind = overall_initial_data.loc[:, outcomes_names].values
    dist_summary = extract_mu_ci_from_summary_accuracy(stan_distributions['overall'],
                                                       ind_cond=[elt for elt in range(len(outcomes_names))])
    draw_all_distributions(dist_ind, dist_summary, len(overall_initial_data), num_cond=len(outcomes_names),
                           std_val=0.05,
                           fname_save=f'../outputs/{study}/results_{study}/{task_name}/{task_name}{name_option}_hrfar.png',
                           **plot_args)


def get_stan_accuracy_distributions(dataframe, conditons_names, nb_trials, name_task, name_add='',
                                    transform_to_accuracy=True):
    """
    For this version: dataframe should have the columns specified in outcome_names - those columns should represent the
    accuracy (and not the number of success!).
    1) 2 functions are used to extract pretest, postest and cross-sessions dataframes
    2) We had the total number of trials - this should be the same for all conditions and for pre/post/across
    3) We transform accuracy into nb of success - NOTE that for across task this is gonna be the mean between the pre and post
    so that we can use the same number of responses for pre/post/across (compared to masataka's work where the number of
    response is 2*nb_trial_per_session and where the number of success was summed)
    """
    # Our goal is to have summary of 2 days experiment (i.e accuracy mean between pre and post);
    # acc in pre and acc in post
    # Divide in pretest / posttest
    pretest, posttest = get_pre_post_dataframe(dataframe, conditons_names)
    # Get mean data for
    sum_observers = get_overall_dataframe_accuracy(dataframe, conditons_names)

    # Add column of number of trials :
    sum_observers['total_resp'], pretest['total_resp'], posttest['total_resp'] = (nb_trials for i in range(3))

    # Transform accuracy into nb of success:
    if transform_to_accuracy:
        transform_accuracy_to_nb_success([sum_observers, pretest, posttest], conditons_names)

    # Change the total number of across condition
    sum_observers['total_resp'] = int(nb_trials * 2)
    sum_observers = sum_observers.astype('int')
    # sum_observers.to_csv('../outputs/v1_ubx/sumdata_' + name_task + name_add + '.csv', header=True, index=False)
    # Compute stan_accuracy for all conditions:
    class_stan_accuracy_overall = [CalStan_accuracy(sum_observers, ind_corr_resp=n) for n in conditons_names]
    class_stan_accuracy_pretest = [CalStan_accuracy(pretest, ind_corr_resp=n) for n in conditons_names]
    class_stan_accuracy_posttest = [CalStan_accuracy(posttest, ind_corr_resp=n) for n in conditons_names]

    # Group all stan distributions into a dict:
    stan_distributions = {'overall': class_stan_accuracy_overall,
                          'pretest': class_stan_accuracy_pretest,
                          'posttest': class_stan_accuracy_posttest}
    return stan_distributions


def transform_accuracy_to_nb_success(dataframe_list, outcomes_names):
    for df in dataframe_list:
        for col in outcomes_names:
            df[col] = df[col] * df['total_resp']


def get_pre_post_dataframe(dataframe, outcomes_names):
    # Divide in pre_test and post_test
    pretest = dataframe[dataframe['task_status'] == 'PRE_TEST'][outcomes_names]
    posttest = dataframe[dataframe['task_status'] == 'POST_TEST'][outcomes_names]
    return pretest, posttest


def get_overall_dataframe_accuracy(dataframe, outcomes_names):
    # summarize two days experiments
    indices_id = extract_id(dataframe, num_count=2)
    sum_observers = []
    for ob in indices_id:
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        sum_observers.append([np.sum(tmp_df[index]) for index in outcomes_names])
    sum_observers = pd.DataFrame(sum_observers, columns=outcomes_names)
    return sum_observers


def get_overall_dataframe_rt(dataframe, outcomes_names):
    # summarize two days experiments
    indices_id = extract_id(dataframe, num_count=2)
    sum_observers = []
    for ob in indices_id:
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        sum_observers.append([np.mean(tmp_df[index]) for index in outcomes_names])
    sum_observers = pd.DataFrame(sum_observers, columns=outcomes_names)
    return sum_observers


def transform_str_to_list(row, columns):
    for column in columns:
        if column in row:
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


def extract_id(dataframe, num_count):
    """
    returns: List of all participants_id
    """
    mask = pd.DataFrame(dataframe.participant_id.value_counts() == num_count)
    indices_id = mask[mask['participant_id'] == True].index.tolist()
    return indices_id


def draw_all_distributions(dist_ind, dist_summary, num_dist, num_cond, std_val=0.05, list_xlim=[0.75, 5.25],
                           list_ylim=[0, 1],
                           list_set_xticklabels=['5', '6', '7', '8', '9'], list_set_xticks=[1, 2, 3, 4, 5],
                           list_set_yticklabels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                           list_set_yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                           val_ticks=0.02,
                           fname_save='workingmemory_accuracy.png',
                           scale_jitter=5,
                           scale_panel=1):
    # dist_ind is the matrix of (num_observers x num_conditions)
    # dist_summary is the mu (numb_conditions), ci_min, and ci_max
    x = [j for j in range(num_cond) for t in range(num_dist)]
    y = dist_ind.T.flatten()
    x_sum = np.linspace(0, num_cond - 1, num_cond)

    df = pd.DataFrame(y)
    df['x'] = x

    fig = plt.figure(figsize=(6.5, 6.5))
    axes = fig.add_subplot(1, scale_panel, 1)
    # axes.scatter(x, dist_ind, s=10, c='blue')
    axes.errorbar(x_sum, dist_summary[:, 0],
                  yerr=[dist_summary[:, 0] - dist_summary[:, 1], dist_summary[:, 2] - dist_summary[:, 0]], capsize=15,
                  fmt='s', markersize=11, ecolor='red', markeredgecolor="black", color='red', mew=1,
                  zorder=num_dist + 2)
    [axes.plot(x_sum, dist_ind[i, :], '-', c='green', linewidth=0.2) for i in range(num_dist)]
    sns.stripplot(x='x', y=0, data=df, ax=axes, color='blue', jitter=scale_jitter / 5., zorder=num_dist + 1)
    axes.set_xlim(list_xlim)
    axes.set_ylim(list_ylim)
    # axes.xaxis.grid(True, which='minor')
    axes.set_xticks(list_set_xticks)
    axes.set_xticklabels(list_set_xticklabels, fontsize=20)
    axes.set_yticks(list_set_yticks)
    axes.set_yticklabels(list_set_yticklabels, fontsize=20)
    # axes.grid()
    axes.yaxis.set_minor_locator(MultipleLocator(val_ticks))
    fig.savefig(fname_save)
    plt.show()


def plot_prepost_mean_accuracy_distribution(conditions, stan_distributions, figname):
    """
    conditions = ["1", "4", "8"]
    stan_distributions is a dict with keys 'overall', 'pretest', 'posttest' with values corresponding to list
    of pystan object where each pystan object is the distribution of p across observers for each condition
    ex : stan_distributions['overall'][index_condition].df_results['theta_accross_obs'] = list of sampled p
    """

    # fig, axs = plt.subplots(len(conditions), 2)
    # if len(conditions) == 1:
    #    axs = np.expand_dims(axs, axis=0)
    fig, axs = plt.subplots(5, 2)

    for index, condition in enumerate(conditions):
        # Get all data:

        tmp_overall = pd.DataFrame(stan_distributions['overall'][index].df_results['theta_across_obs'])
        tmp_overall['condition'] = 'overall'

        tmp_pretest = pd.DataFrame(stan_distributions['pretest'][index].df_results['theta_across_obs'])
        tmp_pretest['condition'] = 'pretest'

        tmp_posttest = pd.DataFrame(stan_distributions['posttest'][index].df_results['theta_across_obs'])
        tmp_posttest['condition'] = 'posttest'

        tmp_diff = tmp_posttest['theta_across_obs'] - tmp_pretest['theta_across_obs']
        df = pd.concat([tmp_overall, tmp_posttest, tmp_pretest], axis=0, ignore_index=True)

        # Now plot correctly what we need:
        sns.histplot(df, x='theta_across_obs', hue='condition', stat='density', kde=True, ax=axs[index, 0])
        axs[index, 0].legend([], [], frameon=False)
        # Put a legend to the right side
        sns.histplot(pd.DataFrame(tmp_diff), stat='probability', kde=True, ax=axs[index, 1])
        axs[index, 1].axvline(x=tmp_diff.mean(), c='red')
        axs[index, 1].axvline(x=np.percentile(tmp_diff.to_list(), 2.5), c='red', linestyle='--')
        axs[index, 1].axvline(x=np.percentile(tmp_diff.to_list(), 97.5), c='red', linestyle='--')
        axs[index, 1].legend([], [], frameon=False)
        axs[index, 1].set_xlim(-0.2, 0.2)
        axs[index, 1].set_ylim(0, 0.035)

    fig.tight_layout()
    fig.savefig(figname)


def plot_prepost_stan_distribution(conditions, stan_distributions, figname, variable):
    """
    conditions = ["1", "4", "8"]
    stan_distributions is a dict with keys 'overall', 'pretest', 'posttest' with values corresponding to list
    of pystan object where each pystan object is the distribution of p across observers for each condition
    ex : stan_distributions['overall'][index_condition].df_results['theta_accross_obs'] = list of sampled p
    """

    # fig, axs = plt.subplots(len(conditions), 2)
    # if len(conditions) == 1:
    #    axs = np.expand_dims(axs, axis=0)
    fig, axs = plt.subplots(5, 2)  # for paper figures.

    for index, condition in enumerate(conditions):
        # Get all data:

        tmp_overall = pd.DataFrame(stan_distributions['overall'][index].df_results[variable])
        tmp_overall['condition'] = 'overall'

        tmp_pretest = pd.DataFrame(stan_distributions['pretest'][index].df_results[variable])
        tmp_pretest['condition'] = 'pretest'

        tmp_posttest = pd.DataFrame(stan_distributions['posttest'][index].df_results[variable])
        tmp_posttest['condition'] = 'posttest'

        tmp_diff = tmp_pretest[variable] - tmp_posttest[variable]
        df = pd.concat([tmp_overall, tmp_posttest, tmp_pretest], axis=0, ignore_index=True)

        # Now plot correctly what we need:
        sns.histplot(df, x=variable, hue='condition', stat='density', kde=True, ax=axs[index, 0])
        axs[index, 0].legend([], [], frameon=False)
        # Put a legend to the right side
        sns.histplot(pd.DataFrame(tmp_diff), stat='probability', kde=True, ax=axs[index, 1])
        axs[index, 1].axvline(x=tmp_diff.mean(), c='red')
        axs[index, 1].axvline(x=np.percentile(tmp_diff.to_list(), 2.5), c='red', linestyle='--')
        axs[index, 1].axvline(x=np.percentile(tmp_diff.to_list(), 97.5), c='red', linestyle='--')
        axs[index, 1].legend([], [], frameon=False)
        axs[index, 1].set_xlim(-200, 200)
        axs[index, 1].set_ylim(0, 0.035)

    fig.tight_layout()
    fig.savefig(figname)
