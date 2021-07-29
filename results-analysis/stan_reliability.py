# %% cell 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stan
import asyncio
import seaborn as sns
from utils import *

asyncio.run(asyncio.sleep(1))


# fname_csv1 = 'accuracy_data.csv'
class CalStan_accuracy():
    def __init__(self, dataframe, ind_corr_resp='corr_resp', ind_total_resp='total_resp', num_chains=4,
                 num_samples=1000):
        self.binomial_code = """
        data {
          int nums;  //total number of participants
          int corr_resp[nums];  //correct response distributions
          int total_resp[nums]; //total trials
        }
        parameters {
          real<lower=0, upper=1> theta[nums]; //correct probability
        }

        model {
          //model
          for (n in 1:nums){
            corr_resp[n] ~ binomial(total_resp[n],theta[n]);
            //the number of correct response depends on the binomial dist.
          }

          //priors
          theta ~ uniform(0,1);
        }

        generated quantities{
          real theta_across_obs=0;
          for (n in 1:nums) {
            theta_across_obs = theta_across_obs + theta[n]/nums;
            } 
        }
        """
        # df_read = pd.read_csv(fname_csv1)
        tmp = [int(val) for val in dataframe[ind_corr_resp].tolist()]
        self.binomial_data = {"nums": len(dataframe),
                              "corr_resp": tmp,
                              "total_resp": dataframe[ind_total_resp].tolist()
                              }
        print(self.binomial_data)
        self.posterior = stan.build(self.binomial_code, data=self.binomial_data)
        self.fit = self.posterior.sample(num_chains=num_chains, num_samples=num_samples)
        # corr_resp = self.fit["corr_resp"]  # array with shape (8, 4000)
        self.df_results = self.fit.to_frame()  # pandas `DataFrame, requires pandas
        # plt.hist(df.loc[:,'diff_theta'])
        self.mu_theta = np.mean(self.df_results.loc[:, 'theta_across_obs'].values)
        self.ci_min = np.percentile(self.df_results.loc[:, 'theta_across_obs'], 2.5)
        self.ci_max = np.percentile(self.df_results.loc[:, 'theta_across_obs'], 97.5)

    def plot_accuracy_per_participant(self):
        tmp = self.df_results[[elt for elt in self.df_results if "theta" in elt]]
        values, participants = [], []
        for index, row in tmp.iterrows():
            for colname, value in row.iteritems():
                values.append(value)
                participants.append(colname)
        df = pd.DataFrame({'participant': participants, 'values': values})
        sns.displot(df, x='values', hue='participant', kind='kde')
        plt.show()


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
        row[f"{key}-RT"] = np.mean(dict_mean_rt_per_condition[key])
        row[f"{key}-accuracy"] = np.mean(dict_mean_accuracy_per_condition[key])
    return row


def plot_prepost_mean_accuracy_distribution(conditions, stan_distributions, figname):
    """
    conditions = ["1", "4", "8"]
    stan_distributions is a dict with keys 'overall', 'pretest', 'posttest' with values corresponding to list
    of pystan object where each pystan object is the distribution of p across observers for each condition
    ex : stan_distributions['overall'][index_condition].df_results['theta_accross_obs'] = list of sampled p
    """

    fig, axs = plt.subplots(len(conditions), 2)

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


if __name__ == '__main__':
    csv_path = "../outputs/moteval/moteval.csv"
    dataframe = pd.read_csv(csv_path)
    dataframe = dataframe.apply(lambda row: transform_str_to_list(row, [
        'results_responses', 'results_rt', 'results_speed_stim', 'results_correct']), axis=1)
    dataframe = delete_uncomplete_participants(dataframe)
    dataframe = dataframe.apply(compute_mean_per_condition, axis=1)

    outcomes_names = ["1-accuracy", "4-accuracy", "8-accuracy"]
    nb_trials = len(dataframe['results_correct'][0])
    # Our goal is to have summary of 2 days experiment (i.e accuracy mean between pre and post);
    # acc in pre and acc in post


    # summarize two days experiments
    indices_id = extract_id(dataframe, num_count=2)
    sum_observers = []
    for ob in indices_id:
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        sum_observers.append([np.mean(tmp_df[index]) for index in outcomes_names])

    sum_observers = pd.DataFrame(sum_observers, columns=outcomes_names)

    sum_observers['total_resp'], pretest['total_resp'], posttest['total_resp'] = (nb_trials for i in range(3))

    for col in outcomes_names:
        sum_observers[col] = sum_observers[col] * sum_observers['total_resp']
        pretest[col] = pretest[col] * pretest['total_resp']
        posttest[col] = posttest[col] * posttest['total_resp']

    class_stan_accuracy_overall = [CalStan_accuracy(sum_observers, ind_corr_resp=n) for n in
                                   [f"{elt}-accuracy" for elt in [1, 4, 8]]]

    class_stan_pretest = [CalStan_accuracy(pretest, ind_corr_resp=n) for n in
                          [f"{elt}-accuracy" for elt in [1, 4, 8]]]

    class_stan_posttest = [CalStan_accuracy(posttest, ind_corr_resp=n) for n in
                           [f"{elt}-accuracy" for elt in [1, 4, 8]]]

    stan_distributions = {'overall': class_stan_accuracy_overall,
                          'pretest': class_stan_pretest,
                          'posttest': class_stan_posttest}

    plot_prepost_mean_accuracy_distribution(outcomes_names, stan_distributions, 'moteval')
