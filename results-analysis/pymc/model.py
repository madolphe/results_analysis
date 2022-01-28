import pymc3 as pm
import arviz as az
import pickle
import matplotlib.pyplot as plt
import os


class PooledModel:
    def __init__(self, data, name, stim_cond_list, sample_size, traces_path=None):
        self.data = data
        self.traces = self.load_trace(traces_path)
        self.name = name
        self.sample_size = sample_size
        if f"{self.name}_results" not in os.listdir():
            os.mkdir(f"{self.name}_results")
        self.stim_condition_list = stim_cond_list
        self.condition = None

    def run(self):
        for condition in self.stim_condition_list:
            self.condition = condition
            self.get_trace()
            self.plot_estimated_posteriors()
            self.plot_CV_figures()
            self.save_trace()

    def describe_data(self):
        print(self.data.describe())

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
            diff_of_means = pm.Deterministic("difference_of_means", post_test_theta-pre_test_theta)
            self.traces = pm.sample(self.sample_size, return_inferencedata=True)

    def get_data_status(self, status: str):
        return self.data[self.data['task_status'] == status]

    def plot_estimated_posteriors(self):
        az.plot_posterior(
            self.traces,
            color="#87ceeb",
        )
        plt.savefig(f"{self.name}_results/posteriors-{self.name}-{self.condition}")
        plt.close()

    def plot_CV_figures(self):
        az.plot_trace(self.traces)
        plt.savefig(f"{self.name}_results/CV-trace-{self.name}-{self.condition}")
        az.plot_forest(self.traces)
        plt.savefig(f"{self.name}_results/CV-forest-{self.name}-{self.condition}")
        az.plot_energy(self.traces)
        plt.savefig(f"{self.name}_results/CV-energy-{self.name}-{self.condition}")
        plt.close()

    def save_trace(self):
        with open(f"{self.name}_results/{self.name}-{self.condition}-trace", 'wb') as buff:
            pickle.dump({'traces': self.traces}, buff)

    @staticmethod
    def load_trace(path):
        if path:
            with open(path, 'rb') as buff:
                data = pickle.load(buff)
                return data['traces']
        else:
            return None

    def compare_traces(self, compared_traces, param_name):
        diff = self.traces.posterior[param_name] - compared_traces.posterior[param_name]
        az.plot_trace(diff)
        az.plot_posterior(diff)
        plt.savefig(f"{self.name}_results/{param_name}-compare-traces-{self.name}.png")


class UnpooledModel(PooledModel):

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


class RTPooledModel(PooledModel):
    def get_trace(self):
        pass

