# https://docs.pymc.io/en/v3/pymc-examples/examples/diagnostics_and_criticism/Bayes_factor.html
# http://www.math.wm.edu/~leemis/chart/UDR/PDFs/StandarduniformStandardtriangular.pdf
import pymc3 as pm
import arviz as az
import pickle
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import copy
import time
import seaborn as sns
from matplotlib.ticker import MultipleLocator


class PooledModel:
    def __init__(self, data, name, group, stim_cond_list, sample_size, folder=".", traces_path=None):
        self.data = data
        self.dict_traces = {}
        self.traces = self.load_trace(traces_path)
        self.name = name
        self.group = group
        self.sample_size = sample_size
        self.folder = folder
        if self.folder not in os.listdir("../outputs/"):
            os.mkdir(f"../outputs/{self.folder}")
        if self.name not in os.listdir(f"../outputs/{self.folder}"):
            os.mkdir(f"../outputs/{self.folder}/{self.name}")
        if f"{self.name}_{self.group}_results" not in os.listdir(f"../outputs/{self.folder}/{self.name}"):
            os.mkdir(f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results")
        if "CV_graph" not in os.listdir(f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results"):
            os.mkdir(f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results/CV_graph")
        if "traces" not in os.listdir(f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results"):
            os.mkdir(f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results/traces")
        if "summary_csv" not in os.listdir(f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results"):
            os.mkdir(f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results/summary_csv")
        self.stim_condition_list = stim_cond_list
        self.condition = None
        self.time = time.time()

    def load_trace(self, paths: dict):
        last_trace = None
        if paths:
            for trace_name, path in paths.items():
                with open(path, 'rb') as buff:
                    data = pickle.load(buff)
                    self.dict_traces[trace_name] = last_trace = data['traces']
        return last_trace

    def run(self, rope=(-0.01, 0.01)):
        for condition in self.stim_condition_list:
            self.condition = condition
            print(f'Start sampling for condition {condition}')
            self.time = time.time()
            self.get_trace()
            print(f'Finish sampling, time taken: {time.time() - self.time}')
            self.time = time.time()
            print('Start plotting figures')
            self.get_figures(rope=(-0.1, 0.1))
            print(f'Finish plotting figures, time elapsed: {time.time() - self.time}')
            az.summary(self.traces).to_csv(
                f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results/summary_csv/summary-{self.name}-{self.condition}.csv")
        self.plot_summary_custom_estimated_posteriors()

    def describe_data(self):
        print(self.data.describe())

    def get_trace(self):
        pre_test = self.get_data_status('PRE_TEST')
        post_test = self.get_data_status('POST_TEST')
        total_success_pre_test = pre_test[f'{self.condition}-correct'].sum()
        total_success_post_test = post_test[f'{self.condition}-correct'].sum()
        with pm.Model() as model:
            beta_dist_pre_test = pm.distributions.continuous.Beta(name="pre_test_posterior",
                                                                  alpha=1 + total_success_pre_test,
                                                                  beta=pre_test[
                                                                           f'{self.condition}-nb'].sum() - total_success_pre_test + 1)
            beta_dist_post_test = pm.distributions.continuous.Beta(name="post_test_posterior",
                                                                   alpha=1 + total_success_post_test,
                                                                   beta=post_test[
                                                                            f'{self.condition}-nb'].sum() - total_success_post_test + 1)
            diff_of_means = pm.Deterministic("difference_of_means", beta_dist_post_test - beta_dist_pre_test)
            self.traces = pm.sample(self.sample_size, cores=1, chains=4)
            self.dict_traces[self.condition] = self.traces

    def get_data_status(self, status: str):
        return self.data[self.data['task_status'] == status]

    def compare_traces(self, compared_traces, param_name):
        diff = self.traces.posterior[param_name] - compared_traces.posterior[param_name]
        return az.summary(diff)
        # az.plot_trace(diff)
        # az.plot_posterior(diff)
        # plt.savefig(f"{self.name}_{self.group}_results/{param_name}-compare-traces-{self.name}.png")

    # # SAVE AND PLOTS FUNCTIONS # #
    def get_figures(self, rope=(-0.01, 0.01)):
        self.plot_estimated_posteriors(rope=(-0.01, 0.01))
        self.plot_CV_figures()
        self.save_trace()
        self.get_infos()

    def plot_estimated_posteriors(self, rope=(-0.01, 0.01)):
        az.plot_posterior(
            self.traces,
            color="#87ceeb",
            rope={'difference_of_means': [{'rope': rope}]}
        )
        plt.savefig(
            f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results/posteriors-{self.name}_{self.group}-{self.condition}")
        plt.close()

    def plot_posterior_and_population(self):
        status_traces = ["pre_test_posterior", "post_test_posterior", "difference_of_means"]
        status_participants = ["PRE_TEST", "POST_TEST", "difference_of_means"]
        self.plot_posterior_and_population_base(status_participants, status_traces)

    def plot_posterior_and_population_base(self, status_participants, status_traces, min_y=0, max_y=1.2,
                                           min_diff_y=-0.5, max_diff_y=0.6, y_tick_step=0.1):
        means, hdis_3, hdis_97 = [], [], []
        for component_participant, component_trace in zip(status_participants, status_traces):
            plt.close()
            y_sig = []
            for index, (condition, traces) in enumerate(self.dict_traces.items()):
                # First plot participants data for each condition
                if component_participant in self.data['task_status'].unique():
                    y = self.data[self.data['task_status'] == component_participant]
                    if f"{condition}-accuracy" in y.keys():
                        y = y[f"{condition}-accuracy"]
                        x = [index] * len(y)
                        plt.scatter(x=x, y=y, marker='*', c='red', s=4)
                # Then store hdi for each condition
                summary = az.summary(traces).loc[[component_trace]]
                means.append(summary['mean'].values[0])
                hdis_3.append(means[-1] - summary['hdi_3%'].values[0])
                hdis_97.append(summary['hdi_97%'].values[0] - means[-1])
                if component_participant == 'difference_of_means':
                    if summary['hdi_3%'].values[0] > 0:
                        y_sig.append(index)
            plt.errorbar(x=[i for i in range(len(means))], y=means, yerr=[hdis_3, hdis_97], fmt='s', markersize=6,
                         capsize=20, c='black')
            plt.plot([i for i in range(len(means))], means, c='black')
            x_ticks = self.dict_traces.keys()
            if component_participant == 'difference_of_means':
                min_y, max_y = min_diff_y, max_diff_y
                plt.axhline(y=0, c='red', linestyle='dotted')
                x_ticks = [condition if index not in y_sig else f"{condition}\n (*)" for index, condition in
                           enumerate(self.dict_traces.keys())]
            plt.yticks(np.arange(min_y, max_y, y_tick_step))
            plt.xticks(ticks=[i for i in range(len(means))], labels=x_ticks)
            plt.title(f"Bayesian estimation \n Task: {self.name} - {component_participant}")
            plt.savefig(
                f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results/posterior-population-{self.name}_{self.group}")
            # plt.show()
            means, hdis_3, hdis_97 = [], [], []

    def plot_comparison_posterior_and_population(self, model_to_compare):
        status_traces = ["pre_test_posterior", "post_test_posterior", "difference_of_means"]
        status_participants = ["PRE_TEST", "POST_TEST", "difference_of_means"]
        self.plot_comparison_posterior_and_population_base(model_to_compare, status_traces, status_participants, )

    def plot_comparison_posterior_and_population_base(self, model_to_compare, status_traces, status_participants,
                                                      x_ticks_min=0, x_ticks_max=1.1,
                                                      x_ticks_step=0.1, x_lim_min=0, x_lim_max=1, x_ticks_min_diff=0,
                                                      x_ticks_max_diff=1.1, x_ticks_step_diff=0.1, x_lim_min_diff=0,
                                                      x_lim_max_diff=1):
        """
        Dict traces of model to compare should contain same parameters estimations
        """
        heights = [i for i in range(1, len(self.dict_traces.keys()) + 1)]
        for component_participant, component_trace in zip(status_participants, status_traces):
            plt.close()
            means, hdis_3, hdis_97 = [], [], []
            means_ref, hdis_3_ref, hdis_97_ref = [], [], []
            for index, ((condition, traces), (condition_ref, traces_ref)) in enumerate(
                    zip(self.dict_traces.items(), model_to_compare.dict_traces.items())):
                background = "white"
                if (index % 2) == 0:
                    background = "silver"
                plt.axhspan(ymin=heights[index] - 0.5, ymax=heights[index] + 0.5, facecolor=background, alpha=0.3)
                if component_participant in self.data['task_status'].unique():
                    # First plot participants data for each condition
                    y = self.data[self.data['task_status'] == component_participant]
                    y_ref = model_to_compare.data[model_to_compare.data['task_status'] == component_participant]
                    if f"{condition}-accuracy" in y.keys():
                        y = y[f"{condition}-accuracy"]
                        y_ref = y_ref[f"{condition}-accuracy"]
                        x = [index] * len(y)
                        plt.scatter(x=y, y=[heights[index] - 0.2] * len(y), marker='*', c='blue', s=4)
                        plt.scatter(x=y_ref, y=[heights[index] + 0.2] * len(y_ref), marker='*', c='red', s=4)
                # Then store hdi for each condition
                try:
                    summary, summary_ref = az.summary(traces).loc[[component_trace]], az.summary(traces_ref).loc[
                        [component_trace]]
                except Exception:
                    print("aaah")
                means.append(summary['mean'].values[0])
                means_ref.append(summary_ref['mean'].values[0])
                hdis_3.append(means[-1] - summary['hdi_3%'].values[0])
                hdis_3_ref.append(means_ref[-1] - summary_ref['hdi_3%'].values[0])
                hdis_97.append(summary['hdi_97%'].values[0] - means[-1])
                hdis_97_ref.append(summary_ref['hdi_97%'].values[0] - means_ref[-1])
            # means and hdi have been completed, now plot errorbars
            plt.errorbar(x=means, y=[height + 0.2 for height in heights], xerr=[hdis_3, hdis_97], fmt='s', markersize=4,
                         capsize=8, c='red')
            plt.errorbar(x=means_ref, y=[height - 0.2 for height in heights], xerr=[hdis_3_ref, hdis_97_ref], fmt='s',
                         markersize=4, capsize=8, c='blue')
            x_ticks = np.arange(x_ticks_min, x_ticks_max, x_ticks_step)
            plt.xlim(x_lim_min, x_lim_max)
            if component_participant == "difference_of_means":
                x_ticks = np.arange(x_ticks_min_diff, x_ticks_max_diff, x_ticks_step_diff)
                plt.xlim(x_lim_min_diff, x_lim_max_diff)
                plt.axvline(x=0, c='red', linestyle='dotted')
            plt.xticks(x_ticks)
            plt.yticks(heights, self.dict_traces.keys())
            plt.ylim(heights[0] - 0.5, heights[-1] + 0.5)
            plt.title(f"{self.name} - {component_participant}")
            plt.savefig(
                f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results/comparison-{self.name}_{self.group}-{self.condition}")

    def plot_summary_custom_estimated_posteriors(self):
        self.plot_summary_custom_estimated_posteriors_base()

    def plot_summary_custom_estimated_posteriors_base(self, x_lim_min=-0.2, x_lim_max=0.2, y_lim_min=0,
                                                      y_lim_max=0.035):
        fig, axs = plt.subplots(len(self.stim_condition_list), 2)
        # Just in case there is only one row, we make sure the row is still callable with [index==0, col]
        if axs.ndim == 1:
            axs = np.expand_dims(axs, axis=0)
        for index, (condition, traces) in enumerate(self.dict_traces.items()):
            # Get all data:
            tmp_pretest = {}
            tmp_pretest['accuracy_samples'] = np.array(traces['pre_test_posterior'])
            tmp_pretest['condition'] = 'pretest'
            tmp_posttest = {}
            tmp_posttest['accuracy_samples'] = np.array(traces['post_test_posterior'])
            tmp_posttest['condition'] = 'posttest'
            tmp_diff = {}
            tmp_diff['diff_samples'] = np.array(traces['difference_of_means'])
            tmp_diff = pd.DataFrame(tmp_diff)
            # Dans df on a besoin de 2 colonnes: 1 pretest/postest et 2 des valeurs de thetas samples
            df = pd.concat([pd.DataFrame(tmp_posttest), pd.DataFrame(tmp_pretest)], axis=0, ignore_index=True)
            # Now plot correctly what we need:
            sns.histplot(df, x='accuracy_samples', hue='condition', stat='density', kde=True, ax=axs[index, 0])
            axs[index, 0].legend([], [], frameon=False)
            # Put a legend to the right side
            # pd.DataFrame(tmp_diff) c'est une colonne avec les valeurs de diff
            sns.histplot(tmp_diff, stat='probability', kde=True, ax=axs[index, 1])
            axs[index, 1].axvline(x=tmp_diff.mean(axis=0).values[0], c='red')
            axs[index, 1].axvline(x=np.percentile(tmp_diff.values, 2.5), c='red', linestyle='--')
            axs[index, 1].axvline(x=np.percentile(tmp_diff.values, 97.5), c='red', linestyle='--')
            axs[index, 1].legend([], [], frameon=False)
            axs[index, 1].set_xlim(x_lim_min, x_lim_max)
            axs[index, 1].set_ylim(y_lim_min, y_lim_max)
        fig.tight_layout()
        fig.savefig(
            f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results/summary-custom-posteriors-{self.name}_{self.group}-{self.condition}")

    def plot_CV_figures(self):
        az.plot_trace(self.traces)
        plt.savefig(
            f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results/CV_graph/CV-trace-{self.name}_{self.group}-{self.condition}")
        az.plot_forest(self.traces)
        plt.savefig(
            f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results/CV_graph/CV-forest-{self.name}_{self.group}-{self.condition}")
        az.plot_energy(self.traces)
        plt.savefig(
            f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results/CV_graph/CV-energy-{self.name}_{self.group}-{self.condition}")
        plt.close()

    def save_trace(self):
        with open(
                f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results/traces/{self.name}_{self.group}-{self.condition}-trace",
                'wb') as buff:
            pickle.dump({'traces': self.traces}, buff)

    def get_infos(self):
        rope = [-0.1, 0.1]
        hdi = az.hdi(self.traces, hdi_prob=0.95)['difference_of_means'].values  # the 95% HDI interval of the difference
        summary = az.summary(self.traces)
        summary['ROPE_in_HDI'] = (rope[1] >= hdi[0]) or (rope[0] <= hdi[1])
        summary.to_csv(
            f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results/summary_csv/{self.condition}-infos.csv")

    @staticmethod
    def compute_effect_size(traces, param, prior_odds):
        traces = traces.posterior[param].values
        BF_values = []
        for chain in range(4):
            nb_sample = len(traces[chain, :])
            nb_in_rope = ((traces[chain, :] > -0.01) & (traces[chain, :] < 0.01)).sum()
            nb_out_rope = (nb_sample - nb_in_rope)
            if nb_in_rope == 0:
                nb_in_rope = 1
            posterior_odds = nb_out_rope / nb_in_rope
            BF_values.append(posterior_odds / prior_odds)
        return BF_values


class PooledModelSimulations(PooledModel):
    def get_trace(self):
        pre_test = self.get_data_status('PRE_TEST')
        post_test = self.get_data_status('POST_TEST')
        with pm.Model() as pooled_model:
            # Independent parameters for each participant
            # total_resp = pm.Data("total_resp", pre_test[f'{self.condition}-nb'].values)
            # prior on theta:
            pre_test_theta = pm.Uniform(name="pre_test_theta", lower=0, upper=1)
            post_test_theta = pm.Uniform(name="post_test_theta", lower=0, upper=1)
            # likelihood
            pre_test_binom = pm.Binomial(name="pre_test_binomial", p=pre_test_theta,
                                         n=pre_test[f'{self.condition}-nb'].values,
                                         observed=pre_test[f'{self.condition}-correct'])
            post_test_binom = pm.Binomial(name="post_test_binomial", p=post_test_theta,
                                          n=post_test[f'{self.condition}-nb'].values,
                                          observed=post_test[f'{self.condition}-correct'])
            diff_of_means = pm.Deterministic("difference_of_means", post_test_theta - pre_test_theta)
            self.traces = pm.sample(self.sample_size, return_inferencedata=True)


class UnpooledModelSimulations(PooledModel):

    def get_trace(self):
        pre_test = self.get_data_status('PRE_TEST')
        post_test = self.get_data_status('POST_TEST')
        with pm.Model() as unpooled_model:
            # Independent parameters for each participant
            # total_resp_pre = pm.Data("total_resp_pre", pre_test[f'{self.condition}-nb'].values)
            # total_resp_post = pm.Data("total_resp_post", post_test[f'{self.condition}-nb'].values)
            # prior on theta:
            pre_test_theta = pm.Uniform(name="pre_test_theta", lower=0.1, upper=1, shape=len(pre_test))
            post_test_theta = pm.Uniform(name="post_test_theta", lower=0.1, upper=1, shape=len(post_test))
            # likelihood
            pre_test_binom = pm.Binomial(name="pre_test_binomial",
                                         p=pre_test_theta,
                                         n=pre_test[f'{self.condition}-nb'],
                                         observed=pre_test[f'{self.condition}-correct'])
            post_test_binom = pm.Binomial(name="post_test_binomial",
                                          p=post_test_theta,
                                          n=post_test[f'{self.condition}-nb'],
                                          observed=post_test[f'{self.condition}-correct'])
            diff_of_means = pm.Deterministic("difference_of_means", post_test_theta - pre_test_theta)
            self.traces = pm.sample(2000, return_inferencedata=True)

    def run(self, rope=(-0.01, 0.01)):
        for condition in self.stim_condition_list:
            self.condition = condition
            self.get_trace()
            self.plot_estimated_posteriors()
            self.plot_CV_figures()
            self.save_trace()
            self.compute_BF()

    def compute_BF(self):
        BF_smc = np.exp(self.traces.report.log_marginal_likelihood)


class PooledModelRTCostSimulations(PooledModel):
    def get_trace(self):
        pre_test = self.get_data_status('PRE_TEST')
        post_test = self.get_data_status('POST_TEST')
        obs_pre_test = pre_test[f"{self.condition}-rt"]
        obs_post_test = post_test[f"{self.condition}-rt"]
        with pm.Model() as pooled_model:
            # prior on mu and sigma:
            # mu_pre_test = pm.Normal('mu_pre_test', mu=0, sd=3)
            # mu_post_test = pm.Normal('mu_post_test', mu=0, sd=3)
            # sigma = pm.HalfNormal('sigma', 3)
            mu_pre_test = pm.Uniform(name='pre_test_posterior', lower=-1400, upper=1400)
            mu_post_test = pm.Uniform(name='post_test_posterior', lower=-1400, upper=1400)
            sigma = pm.Uniform(name='sigma', lower=0, upper=100)
            RT_pred_pre_test = pm.Normal(name='RT_pred_pre_test', mu=mu_pre_test, sd=sigma, observed=obs_pre_test)
            RT_pred_post_test = pm.Normal(name='RT_pred_post_test', mu=mu_post_test, sd=sigma, observed=obs_post_test)
            diff_of_means = pm.Deterministic("difference_of_means", mu_pre_test - mu_post_test)
            # self.traces = pm.sample(self.sample_size, return_inferencedata=True)
            self.traces = pm.sample(self.sample_size)
        self.dict_traces[self.condition] = self.traces

    def plot_posterior_and_population(self):
        status_traces = ["pre_test_posterior", "post_test_posterior", "difference_of_means"]
        status_participants = ["PRE_TEST", "POST_TEST", "difference_of_means"]
        config = {"min_y": -300, "max_y": 300, "min_diff_y": -300, "max_diff_y": 400, "y_tick_step": 100}
        self.plot_posterior_and_population_base(status_participants=status_participants, status_traces=status_traces,
                                                **config)

    def plot_comparison_posterior_and_population(self, model_to_compare):
        config = {"x_ticks_min": -300, "x_ticks_max": 400,
                  "x_ticks_step": 100, "x_lim_min": -300, "x_lim_max": 300, "x_ticks_min_diff": -50,
                  "x_ticks_max_diff": 50, "x_ticks_step_diff": 10, "x_lim_min_diff": -50,
                  "x_lim_max_diff": 50}
        status_traces = ["pre_test_posterior", "post_test_posterior", "difference_of_means"]
        status_participants = ["PRE_TEST", "POST_TEST", "difference_of_means"]
        self.plot_comparison_posterior_and_population_base(model_to_compare, status_participants=status_participants,
                                                           status_traces=status_traces,
                                                           **config)

    def plot_summary_custom_estimated_posteriors(self):
        config = {"x_lim_min": -400, "x_lim_max": 400, "y_lim_min": 0, "y_lim_max": 0.5}
        self.plot_summary_custom_estimated_posteriors_base(**config)


class PooledModelRTSimulations(PooledModel):
    def get_trace(self):
        pre_test = self.get_data_status('PRE_TEST')
        post_test = self.get_data_status('POST_TEST')
        obs_pre_test = pre_test[f"{self.condition}-rt"]
        obs_post_test = post_test[f"{self.condition}-rt"]
        with pm.Model() as pooled_model:
            # prior on mu and sigma:
            # mu_pre_test = pm.Normal('mu_pre_test', mu=0, sd=3)
            # mu_post_test = pm.Normal('mu_post_test', mu=0, sd=3)
            # sigma = pm.HalfNormal('sigma', 3)
            mu_pre_test = pm.Uniform(name='pre_test_posterior', lower=0, upper=1400)
            mu_post_test = pm.Uniform(name='post_test_posterior', lower=0, upper=1400)
            sigma = pm.Uniform(name='sigma', lower=0, upper=1400)
            RT_pred_pre_test = pm.Normal(name='RT_pred_pre_test', mu=mu_pre_test, sd=sigma, observed=obs_pre_test)
            RT_pred_post_test = pm.Normal(name='RT_pred_post_test', mu=mu_post_test, sd=sigma, observed=obs_post_test)
            diff_of_means = pm.Deterministic("difference_of_means", mu_pre_test - mu_post_test)
            self.traces = pm.sample(self.sample_size)
        self.dict_traces[self.condition] = self.traces

    def plot_posterior_and_population(self):
        status_traces = ["pre_test_posterior", "post_test_posterior", "difference_of_means"]
        status_participants = ["PRE_TEST", "POST_TEST", "difference_of_means"]
        config = {"min_y": 0, "max_y": 1500, "min_diff_y": -400, "max_diff_y": 400, "y_tick_step": 100}
        self.plot_posterior_and_population_base(status_participants, status_traces, **config)

    def plot_comparison_posterior_and_population(self, model_to_compare):
        config = {"x_ticks_min": 0, "x_ticks_max": 1400,
                  "x_ticks_step": 100, "x_lim_min": 0, "x_lim_max": 1400, "x_ticks_min_diff": -250,
                  "x_ticks_max_diff": 300, "x_ticks_step_diff": 50, "x_lim_min_diff": -250,
                  "x_lim_max_diff": 250}
        status_traces = ["pre_test_posterior", "post_test_posterior", "difference_of_means"]
        status_participants = ["PRE_TEST", "POST_TEST", "difference_of_means"]
        self.plot_comparison_posterior_and_population_base(model_to_compare, status_traces, status_participants,
                                                           **config)

    def plot_summary_custom_estimated_posteriors(self):
        config = {"x_lim_min": -400, "x_lim_max": 400, "y_lim_min": 0, "y_lim_max": 0.5}
        self.plot_summary_custom_estimated_posteriors_base(**config)


class GLModel(PooledModel):
    def get_trace(self):
        X = pd.get_dummies(self.data['condition'], drop_first=True)
        obs = self.data[self.condition]
        with pm.Model() as LinearModel:
            intercept = pm.Normal(name='baseline', mu=0, sigma=5)
            slope = pm.Normal(name='slope', mu=0, sigma=5)
            mu = intercept + pm.math.dot(X, slope)  # + pm.math.dot(X_week, slop_week)
            # mu = intercept + pm.math.dot(X, slope) + pm.math.dot(X_week, slop_week)
            sig_scores = pm.HalfNormal('sigma', 3)
            scores = pm.Normal(name="scores", mu=mu, sigma=sig_scores, observed=obs)
            print('START SAMPLING')
            self.traces = pm.sample(self.sample_size, return_inferencedata=True)
            print('SAMPLING IS DONE')

    def plot_estimated_posteriors(self, rope=(-0.1, 0.1)):
        az.plot_posterior(
            self.traces,
            color="#87ceeb",
            rope={'baseline': [{'rope': (-0.5, 0.5)}],
                  'slope': [{'rope': (-0.1, 0.1)}]}
        )
        plt.savefig(
            f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results/posteriors-{self.name}_{self.group}-{self.condition}.png")
        plt.close()

    def get_infos(self):
        rope = [-0.1, 0.1]
        hdi = az.hdi(self.traces, hdi_prob=0.95)['slope'].values  # the 95% HDI interval of the difference
        summary = az.summary(self.traces)
        summary['ROPE_in_HDI'] = (rope[1] >= hdi[0]) or (rope[0] <= hdi[1])
        summary.to_csv(
            f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results/{self.condition}-infos.csv")


class PooledModelRTGLMSimulations(PooledModel):
    """
    https://discourse.pymc.io/t/lognormal-model-for-reaction-times-how-to-specify/6677/3
    """

    def get_trace(self):
        pre_test = self.get_data_status('PRE_TEST')
        post_test = self.get_data_status('POST_TEST')
        X_pre_test = pd.get_dummies(pre_test['condition'], drop_first=True)
        X_post_test = pd.get_dummies(post_test['condition'], drop_first=True)
        obs_pre_test = np.log2(pre_test[f"{self.condition}-rt"])
        obs_post_test = np.log2(post_test[f"{self.condition}-rt"])
        with pm.Model() as RT_model:
            baseline_pre_test = pm.Normal('baseline_pre_test', mu=0, sd=3)
            baseline_post_test = pm.Normal('baseline_post_test', mu=0, sd=3)
            treatment_pre_test = pm.Normal('treatment_pre_test', mu=0, sd=1, shape=1)
            treatment_post_test = pm.Normal('treatment_post_test', mu=0, sd=1, shape=1)
            sigma = pm.HalfNormal('sigma', 3)
            mu_pre_test = baseline_pre_test + pm.math.dot(X_pre_test, treatment_pre_test)
            mu_post_test = baseline_post_test + pm.math.dot(X_post_test, treatment_post_test)
            RT_pred_pre_test = pm.Lognormal(name='RT_pred_pre_test', mu=mu_pre_test, sd=sigma, observed=obs_pre_test)
            RT_pred_post_test = pm.Lognormal(name='RT_pred_post_test', mu=mu_post_test, sd=sigma,
                                             observed=obs_post_test)
            prior_predictive = pm.sample_prior_predictive()
            diff_of_means = pm.Deterministic("difference_of_means", RT_pred_pre_test - RT_pred_post_test)
            self.traces = pm.sample(2000, return_inferencedata=True)


class NormalNormalQuestionnaireModel(PooledModel):
    def __init__(self, data, name, group, stim_cond_list, sample_size, session_id_list, folder=".", traces_path=None):
        super(NormalNormalQuestionnaireModel, self).__init__(data, name, group, stim_cond_list, sample_size,
                                                             folder=folder, traces_path=traces_path)
        self.session_id_list = session_id_list
        self.tmp_session_id = None
        self.traces = {}

    def run(self, rope=(-0.01, 0.01)):
        """
        Compute difference between sessions
        """
        for condition in self.stim_condition_list:
            for session_id in self.session_id_list:
                self.condition = condition
                self.tmp_session_id = session_id
                # self.tmp_session_id_1 = self.session_id_list[index_session_id + 1]
                print(f'Start sampling for condition {condition}')
                self.time = time.time()
                self.get_trace()
                print(f'Finish sampling, time taken: {time.time() - self.time}')
                self.time = time.time()
                print('Start plotting figures')
            # All traces are stored, let's keep it in backup_traces
            backup_traces = copy.deepcopy(self.traces)
            backup_condition = self.condition
            for index_session_id, session_id in enumerate(self.session_id_list):
                self.condition = backup_condition + f"-{session_id}"
                self.traces = backup_traces[self.condition]
                self.get_figures(rope=(-0.1, 0.1))
                print(f'Finish plotting figures, time elapsed: {time.time() - self.time}')
                az.summary(self.traces).to_csv(
                    f"../outputs/{self.folder}/{self.name}/{self.name}_{self.group}_results/summary-{self.name}-{self.condition}.csv")
                # I do this after :)
                # Compare pairwise traces:
                # if (index_session_id+1) < len(self.session_id_list):
                #     pairwise_mu_diff = self.compare_traces(
                #         backup_traces[f"{backup_condition}-{self.session_id_list[index_session_id + 1]}"],
                #         param_name=f'mu')
                #     pairwise_sigma_diff = self.compare_traces(
                #         backup_traces[f"{backup_condition}-{self.session_id_list[index_session_id + 1]}"],
                #         param_name=f'sigma')
            # Restore the placeholder for traces
            self.traces = {}

    def get_trace(self):
        obs = self.data.query(f'session_id == {self.tmp_session_id}')
        obs = obs[self.condition]
        with pm.Model():
            mu = pm.Uniform(name=f'mu', lower=0, upper=100)
            sigma = pm.Uniform(name=f'sigma', lower=0, upper=100)
            answer_pop = pm.Normal(name='answer_population', mu=mu, sd=sigma, observed=obs)
            self.traces[f"{self.condition}-{self.tmp_session_id}"] = pm.sample(self.sample_size,
                                                                               return_inferencedata=True)

    # # SAVE AND PLOTS FUNCTIONS # #
    def get_figures(self, rope=(-0.01, 0.01)):
        self.plot_estimated_posteriors(rope=(-0.01, 0.01))
        self.plot_CV_figures()
        self.save_trace()
