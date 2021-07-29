import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cal_stan_accuracy_rt import CalStan_accuracy
from matplotlib.ticker import MultipleLocator


def extract_mu_ci_from_summary_accuracy(dataframe, ind_cond):
    # 3 means the mu, ci_min, and ci_max
    out = np.zeros((len(ind_cond), 3))
    for t, ind in enumerate(ind_cond):
        out[t, 0] = dataframe[ind].mu_theta
        out[t, 1] = dataframe[ind].ci_min
        out[t, 2] = dataframe[ind].ci_max
    return out


def plot_all_accuracy_figures(stan_distributions, outcomes_names, task_name, overall_initial_data, nb_trials):
    plot_prepost_mean_accuracy_distribution(outcomes_names, stan_distributions,
                                            f'../outputs/{task_name}/{task_name}_distrib_reliability.png')
    dist_ind = overall_initial_data.loc[:, outcomes_names].values
    dist_summary = extract_mu_ci_from_summary_accuracy(stan_distributions['overall'],
                                                       ind_cond=[elt for elt in range(len(outcomes_names))])
    draw_all_distributions(dist_ind, dist_summary, len(overall_initial_data), num_cond=len(outcomes_names), std_val=0.05,
                           fname_save=f'../outputs/{task_name}/{task_name}_hrfar.png')


def get_stan_accuracy_distributions(dataframe, outcomes_names, nb_trials):
    # Our goal is to have summary of 2 days experiment (i.e accuracy mean between pre and post);
    # acc in pre and acc in post
    # Divide in pretest / posttest
    pretest, posttest = get_pre_post_dataframe(dataframe, outcomes_names)
    # Get mean data for
    sum_observers = get_overall_dataframe(dataframe, outcomes_names)

    # Add column of number of trials :
    sum_observers['total_resp'], pretest['total_resp'], posttest['total_resp'] = (nb_trials for i in range(3))

    # Transform accuracy into nb of success:
    transform_accuracy_to_nb_success([sum_observers, pretest, posttest], outcomes_names)

    # Compute stan_accuracy for all conditions:
    class_stan_accuracy_overall = [CalStan_accuracy(sum_observers, ind_corr_resp=n) for n in outcomes_names]
    class_stan_accuracy_pretest = [CalStan_accuracy(pretest, ind_corr_resp=n) for n in outcomes_names]
    class_stan_accuracy_posttest = [CalStan_accuracy(posttest, ind_corr_resp=n) for n in outcomes_names]

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


def get_overall_dataframe(dataframe, outcomes_names):
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


def draw_all_distributions(dist_ind,dist_summary,num_dist,num_cond,std_val = 0.05,list_xlim=[0.75,5.25],list_ylim=[0,1],
                                list_set_xticklabels=['5','6','7','8','9'],list_set_xticks=[1,2,3,4,5],
                                list_set_yticklabels=['0.0','0.2','0.4','0.6','0.8','1.0'],list_set_yticks=[0,0.2,0.4,0.6,0.8,1.0],
                                val_ticks = 0.02,
                                fname_save='workingmemory_accuracy.png'):
    # dist_ind is the matrix of (num_observers x num_conditions)
    # dist_summary is the mu (numb_conditions), ci_min, and ci_max
    x = [j+1+std_val*np.random.randn() for j in range(num_cond) for t in range(num_dist)]
    dist_ind = dist_ind.T.flatten()
    x_sum = np.linspace(1,num_cond,num_cond)
    fig = plt.figure(figsize=(5,5))
    axes = fig.add_subplot(1,1,1)
    axes.scatter(x,dist_ind,s=10,c='blue')
    axes.errorbar(x_sum, dist_summary[:,0], yerr = [dist_summary[:,0]-dist_summary[:,1], dist_summary[:,2]-dist_summary[:,0]], capsize=5, fmt='o', markersize=13, ecolor='red', markeredgecolor = "red", color='w')
    axes.set_xlim(list_xlim)
    axes.set_ylim(list_ylim)
    axes.xaxis.grid(True, which='minor')
    axes.set_xticks(list_set_xticks)
    axes.set_xticklabels(list_set_xticklabels,fontsize=20)
    axes.set_yticks(list_set_yticks)
    axes.set_yticklabels(list_set_yticklabels,fontsize=20)
    axes.grid()
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

    fig, axs = plt.subplots(len(conditions), 2)
    if len(conditions) == 1:
        axs = np.expand_dims(axs, axis=0)

    for index, condition in enumerate(conditions):
        # Get all data:

        tmp_overall = pd.DataFrame(stan_distributions['overall'][index].df_results['theta_across_obs'])
        tmp_overall['condition'] = 'overall'

        tmp_pretest = pd.DataFrame(stan_distributions['pretest'][index].df_results['theta_across_obs'])
        tmp_pretest['condition'] = 'pretest'

        tmp_posttest = pd.DataFrame(stan_distributions['posttest'][index].df_results['theta_across_obs'])
        tmp_posttest['condition'] = 'posttest'

        tmp_diff = tmp_posttest['theta_across_obs'] - tmp_pretest['theta_across_obs']
        df = pd.concat([tmp_overall, tmp_posttest, tmp_pretest], axis=0)

        # Now plot correctly what we need:
        sns.histplot(df, x='theta_across_obs', hue='condition', stat='density', kde=True, ax=axs[index, 0])
        axs[index, 0].legend([], [], frameon=False)
        # Put a legend to the right side
        sns.histplot(pd.DataFrame(tmp_diff), stat='density', kde=True, ax=axs[index, 1])
        axs[index, 1].axvline(x=tmp_diff.mean(), c='red')
        axs[index, 1].axvline(x=np.percentile(tmp_diff.to_list(), 2.5), c='red', linestyle='--')
        axs[index, 1].axvline(x=np.percentile(tmp_diff.to_list(), 97.5), c='red', linestyle='--')
        axs[index, 1].legend([], [], frameon=False)

    fig.tight_layout()
    fig.savefig(figname)

