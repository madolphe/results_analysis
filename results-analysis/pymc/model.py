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


class PooledModel:
    def __init__(self, data, name, group, stim_cond_list, sample_size, folder=".", traces_path=None):
        self.data = data
        self.traces = self.load_trace(traces_path)
        self.name = name
        self.group = group
        self.sample_size = sample_size
        self.folder = folder
        if self.folder not in os.listdir():
            os.mkdir(self.folder)
        if self.name not in os.listdir(self.folder):
            os.mkdir(f"{self.folder}/{self.name}")
        if f"{self.name}_{self.group}_results" not in os.listdir(f"{self.folder}/{self.name}"):
            os.mkdir(f"{self.folder}/{self.name}/{self.name}_{self.group}_results")
        self.stim_condition_list = stim_cond_list
        self.condition = None
        self.time = time.time()

    def run(self):
        for condition in self.stim_condition_list:
            self.condition = condition
            print(f'Start sampling for condition {condition}')
            self.time = time.time()
            self.get_trace()
            print(f'Finish sampling, time taken: {time.time() - self.time}')
            self.time = time.time()
            print('Start plotting figures')
            self.get_figures()
            print(f'Finish plotting figures, time elapsed: {time.time() - self.time}')
            az.summary(self.traces).to_csv(
                f"{self.folder}/{self.name}/{self.name}_{self.group}_results/summary-{self.name}-{self.condition}.csv")

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
            self.traces = pm.sample(self.sample_size)

    def get_data_status(self, status: str):
        return self.data[self.data['task_status'] == status]

    def compare_traces(self, compared_traces, param_name):
        diff = self.traces.posterior[param_name] - compared_traces.posterior[param_name]
        return az.summary(diff)
        # az.plot_trace(diff)
        # az.plot_posterior(diff)
        # plt.savefig(f"{self.name}_{self.group}_results/{param_name}-compare-traces-{self.name}.png")

    # # SAVE AND PLOTS FUNCTIONS # #
    def get_figures(self):
        self.plot_estimated_posteriors()
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
            f"{self.folder}/{self.name}/{self.name}_{self.group}_results/posteriors-{self.name}_{self.group}-{self.condition}")
        plt.close()

    def plot_CV_figures(self):
        az.plot_trace(self.traces)
        plt.savefig(
            f"{self.folder}/{self.name}/{self.name}_{self.group}_results/CV-trace-{self.name}_{self.group}-{self.condition}")
        az.plot_forest(self.traces)
        plt.savefig(
            f"{self.folder}/{self.name}/{self.name}_{self.group}_results/CV-forest-{self.name}_{self.group}-{self.condition}")
        az.plot_energy(self.traces)
        plt.savefig(
            f"{self.folder}/{self.name}/{self.name}_{self.group}_results/CV-energy-{self.name}_{self.group}-{self.condition}")
        plt.close()

    def save_trace(self):
        with open(
                f"{self.folder}/{self.name}/{self.name}_{self.group}_results/{self.name}_{self.group}-{self.condition}-trace",
                'wb') as buff:
            pickle.dump({'traces': self.traces}, buff)

    @staticmethod
    def load_trace(path):
        if path:
            with open(path, 'rb') as buff:
                data = pickle.load(buff)
                return data['traces']
        else:
            return None

    def get_infos(self):
        rope = [-0.1, 0.1]
        hdi = az.hdi(self.traces, hdi_prob=0.95)['difference_of_means'].values  # the 95% HDI interval of the difference
        summary = az.summary(self.traces)
        summary['ROPE_in_HDI'] = (rope[1] >= hdi[0]) or (rope[0] <= hdi[1])
        summary.to_csv(f"{self.folder}/{self.name}/{self.name}_{self.group}_results/{self.condition}-infos.csv")

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

    def run(self):
        for condition in self.stim_condition_list:
            self.condition = condition
            self.get_trace()
            self.plot_estimated_posteriors()
            self.plot_CV_figures()
            self.save_trace()
            self.compute_BF()

    def compute_BF(self):
        BF_smc = np.exp(self.traces.report.log_marginal_likelihood)


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


class PooledModelRTSimulations(PooledModel):
    def get_trace(self):
        pre_test = self.get_data_status('PRE_TEST')
        post_test = self.get_data_status('POST_TEST')
        obs_pre_test = np.log2(pre_test[f"{self.condition}-rt"])
        obs_post_test = np.log2(post_test[f"{self.condition}-rt"])
        with pm.Model() as pooled_model:
            # prior on mu and sigma:
            mu_pre_test = pm.Normal('mu_pre_test', mu=0, sd=3)
            mu_post_test = pm.Normal('mu_post_test', mu=0, sd=3)
            sigma = pm.HalfNormal('sigma', 3)
            RT_pred_pre_test = pm.Lognormal(name='RT_pred_pre_test', mu=mu_pre_test, sd=sigma, observed=obs_pre_test)
            RT_pred_post_test = pm.Lognormal(name='RT_pred_post_test', mu=mu_post_test, sd=sigma, observed=obs_post_test)
            diff_of_means = pm.Deterministic("difference_of_means", mu_post_test - mu_pre_test)
            self.traces = pm.sample(self.sample_size, return_inferencedata=True)


class GLModel(PooledModel):
    def get_trace(self):
        X = pd.get_dummies(self.data['condition'], drop_first=True)
        obs = self.data[self.condition]
        with pm.Model() as LinearModel:
            intercept = pm.Normal(name='baseline', mu=0, sigma=5)
            slope = pm.Normal(name='slope', mu=0, sigma=5)
            mu = intercept + pm.math.dot(X, slope)
            sig_scores = pm.HalfNormal('sigma', 3)
            scores = pm.Normal(name="scores", mu=mu, sigma=sig_scores, observed=obs)
            print('START SAMPLING')
            self.traces = pm.sample(self.sample_size, return_inferencedata=True)
            print('SAMPLING IS DONE')

    def plot_estimated_posteriors(self, rope=(-0.1, 0.1)):
        az.plot_posterior(
            self.traces,
            color="#87ceeb",
            rope={'slope': [{'rope': rope}]}
        )
        plt.savefig(
            f"{self.folder}/{self.name}/{self.name}_{self.group}_results/posteriors-{self.name}_{self.group}-{self.condition}")
        plt.close()

    def get_infos(self):
        rope = [-0.1, 0.1]
        hdi = az.hdi(self.traces, hdi_prob=0.95)['slope'].values  # the 95% HDI interval of the difference
        summary = az.summary(self.traces)
        summary['ROPE_in_HDI'] = (rope[1] >= hdi[0]) or (rope[0] <= hdi[1])
        summary.to_csv(f"{self.folder}/{self.name}/{self.name}_{self.group}_results/{self.condition}-infos.csv")