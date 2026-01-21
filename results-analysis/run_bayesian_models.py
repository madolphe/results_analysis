import pymc as pm
import arviz as az
import pandas as pd
import json
import numpy as np
import pdb
from pathlib import Path
import matplotlib
from ast import literal_eval
import os
from sklearn.metrics import ConfusionMatrixDisplay
import math
import scipy.stats as stats

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)
import matplotlib.pyplot as plt


def build_and_get_trace(df, task_condition, model_type, nb_samples, path_to_store, graphviz=False):
    # Reformat data and get a trace as Inference data
    data_combined, coords, deltas, data_pre, data_post = reformat_data(df, task_condition)
    model = build_model(model_type, data_combined, task_condition=task_condition, coords=coords)
    if graphviz:
        gv = pm.model_to_graphviz(model)
        gv.format = 'png'
        gv.render(filename=f"{model_type}-graphviz")
    trace = sample_model(model, nb_samples)
    trace.to_json(f"{path_to_store}-{model_type}_inference_data.json")
    return trace


def reformat_data(df, task_condition):
    # Create 'condition' and 'time' indices
    data_pre = df[df["task_status"] == "PRE_TEST"]
    data_pre['condition_idx'] = (data_pre['condition'] == 'zpdes').astype(int)
    data_pre['time_idx'] = 0  # 0 for 'pre'
    data_post = df[df["task_status"] == "POST_TEST"]
    data_post['condition_idx'] = (data_post['condition'] == 'zpdes').astype(int)
    data_post['time_idx'] = 1  # 1 for 'post'
    data_pre = data_pre.set_index('participant_id')
    data_post = data_post.set_index('participant_id')
    # To handle control condition, let's add a third possibility "control"
    # Merge data_pre and data_post into one dataframe
    data_combined = pd.concat([data_pre, data_post], axis=0)
    if sum(data_combined['condition'] == "no_condition") > 0:
        conditions = ["control"]
    else:
        conditions = ["baseline", "zpdes"]
    times = ["pre", "post"]
    coords = {"condition": conditions, "time": times}
    # For individuals (might be deleted):
    if f"{task_condition}-nb" in df:
        n = data_combined[f"{task_condition}-nb"].unique()[0]
        deltas = (data_post[f'{task_condition}-correct'] - data_pre[f'{task_condition}-correct']) / n
    else:
        deltas = []
    return data_combined, coords, deltas, data_pre, data_post


def build_model(type, data, task_condition=None, coords=None):
    models = {"hierarbinom": build_hierar_binom,
              "hierar_binom_n_var": build_hierar_binom_n_var,
              "RT_normal": build_RT_normal,
              "RT_normal_hyper": build_RT_normal_hyper,
              "ufov_threshold": build_ufov_threshold,
              "ufov_threshold_hyper": build_ufov_threshold_hyper,
              "switching_cost_hyper": build_switching_cost_RT_hyper,
              "switching_cost": build_switching_cost_RT,
              "precision_normal_hyper": build_precision_moteval_hyper,
              "precision_normal": build_precision_normal
              }
    return models[type](data, task_condition, coords)


def sample_model(model, nb):
    with model:
        return pm.sample(nb, idata_kwargs={'log_likelihood': True})


def plot_trace(trace, var_names, path_to_store, model_type="hierarbinom"):
    az.plot_trace(trace, var_names=var_names, compact=True, legend=True)
    az.summary(trace)
    plt.savefig(f"{path_to_store}_traces_{model_type}.png")
    plt.close()


def build_precision_moteval_hyper(data_combined, task_condition, coords):
    with pm.Model(coords=coords) as precision_model:
        # Hyperpriors for population mean and standard deviation
        mu_hyper = pm.Normal('mu_hyper', mu=0.5, sigma=0.1)
        sigma_hyper = pm.HalfNormal('sigma_hyper', 0.1)
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        # Participant-level means
        mu = pm.Normal('mu', mu=mu_hyper, sigma=sigma_hyper, dims=("condition", "time"))
        # Deterministic parameters for the differences
        # If 2 groups:
        # pdb.set_trace()
        if len(coords['condition']) > 1:
            delta_zpdes = pm.Deterministic("delta_zpdes", mu[1, 1] - mu[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", mu[0, 1] - mu[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        # Else, i.e control:
        else:
            delta = pm.Deterministic("delta", mu[0, 1] - mu[0, 0])
        # Common standard deviation for all participants
        sigma = pm.HalfNormal('sigma', 0.1, dims=("condition", "time"))
        # likelihood
        pm.Normal("y", mu=mu[g, t], sigma=sigma[g, t],
                  observed=data_combined[f'{task_condition}-accuracy_continuous'], dims="obs_id")
    return precision_model


def build_precision_normal(data_combined, task_condition, coords):
    with pm.Model(coords=coords) as precision_model:
        # Hyperpriors for population mean and standard deviation
        # mu_hyper = pm.Normal('mu_hyper', mu=0.5, sigma=0.1)
        # sigma_hyper = pm.HalfNormal('sigma_hyper', 0.1)
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        # Participant-level means
        mu = pm.Normal('mu', mu=0.5, sigma=0.1, dims=("condition", "time"))
        # Deterministic parameters for the differences
        # If 2 groups:
        # pdb.set_trace()
        if len(coords['condition']) > 1:
            delta_zpdes = pm.Deterministic("delta_zpdes", mu[1, 1] - mu[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", mu[0, 1] - mu[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        # Else, i.e control:
        else:
            delta = pm.Deterministic("delta", mu[0, 1] - mu[0, 0])
        # Common standard deviation for all participants
        sigma = pm.HalfNormal('sigma', 0.1, dims=("condition", "time"))
        # likelihood
        pm.Normal("y", mu=mu[g, t], sigma=sigma[g, t],
                  observed=data_combined[f'{task_condition}-accuracy_continuous'], dims="obs_id")
    return precision_model


def build_hierar_binom(data_combined, task_condition, coords):
    n = data_combined[f"{task_condition}-nb"].unique()[0]
    with pm.Model(coords=coords) as hierarchical_model:
        # Global-level priors (one for each condition and time)
        beta = pm.Beta("beta", alpha=1, beta=1, dims=("condition", "time"))
        # Deterministic parameters for the differences
        if len(coords['condition']) > 1:
            delta_zpdes = pm.Deterministic("delta_zpdes", beta[1, 1] - beta[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", beta[0, 1] - beta[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        # Else, i.e control:
        else:
            delta = pm.Deterministic("delta", beta[0, 1] - beta[0, 0])
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        # Introduce a variability term for the local-level
        sigma = pm.HalfNormal('sigma', 0.1, dims=("condition", "time"))
        # Model local-level p's using the global-level p's as mean and sigma as SD
        p_observation = pm.Beta('p_observation', mu=beta[g, t], sigma=sigma[g, t], dims="obs_id")
        # Likelihood
        pm.Binomial("y", n=n, p=p_observation, observed=data_combined[f'{task_condition}-correct'].values,
                    dims="obs_id")
    return hierarchical_model


def build_hierar_binom_n_var(data_combined, task_condition, coords):
    with pm.Model(coords=coords) as hierarchical_model:
        # Global-level priors (one for each condition and time)
        beta = pm.Beta("beta", alpha=1, beta=1, dims=("condition", "time"))
        # Deterministic parameters for the differences
        if len(coords['condition']) > 1:
            delta_zpdes = pm.Deterministic("delta_zpdes", beta[1, 1] - beta[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", beta[0, 1] - beta[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        else:
            delta = pm.Deterministic("delta", beta[0, 1] - beta[0, 0])  # post - pre for ctrl
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        n = pm.MutableData("n", data_combined[f'{task_condition}-nb'], dims="obs_id")
        # Introduce a variability term for the local-level
        sigma = pm.HalfNormal('sigma', 0.1, dims=("condition", "time"))
        # Model local-level p's using the global-level p's as mean and sigma as SD
        p_observation = pm.Beta('p_observation', mu=beta[g, t], sigma=sigma[g, t], dims="obs_id")
        # Likelihood
        pm.Binomial("y", n=n, p=p_observation, observed=data_combined[f'{task_condition}-correct'].values,
                    dims="obs_id")
    return hierarchical_model


def build_RT_normal_hyper(data_combined, task_condition, coords):
    with pm.Model(coords=coords) as RT_model:
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        # Hyperpriors for population mean and standard deviation
        mu_hyper = pm.Uniform('mu_hyper', lower=0, upper=3000)
        sigma_hyper = pm.HalfNormal('sigma_hyper', 100)
        # Participant-level means
        mu = pm.Normal('mu', mu=mu_hyper, sigma=sigma_hyper, dims=("condition", "time"))
        # Common standard deviation for all participants
        sigma = pm.HalfNormal('sigma', 0.1, dims=("condition", "time"))
        # Deterministic parameters for the differences
        if len(coords['condition']) > 1:
            delta_zpdes = pm.Deterministic("delta_zpdes", mu[1, 1] - mu[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", mu[0, 1] - mu[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        else:
            delta = pm.Deterministic("delta", mu[0, 1] - mu[0, 0])  # post - pre for zpdes
        # likelihood
        pm.Normal("y", mu=mu[g, t], sigma=sigma[g, t],
                  observed=data_combined[f'{task_condition}-rt'], dims="obs_id")
    return RT_model


def build_RT_normal(data_combined, task_condition, coords):
    with pm.Model(coords=coords) as RT_model:
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        # Hyperpriors for population mean and standard deviation
        # mu_hyper = pm.Uniform('mu_hyper', lower=0, upper=3000)
        # sigma_hyper = pm.HalfNormal('sigma_hyper', 100)
        # Participant-level means
        mu = pm.Normal('mu', mu=500, sigma=50, dims=("condition", "time"))
        # Common standard deviation for all participants
        sigma = pm.HalfNormal('sigma', 10, dims=("condition", "time"))
        # Deterministic parameters for the differences
        if len(coords['condition']) > 1:
            delta_zpdes = pm.Deterministic("delta_zpdes", mu[1, 1] - mu[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", mu[0, 1] - mu[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        else:
            delta = pm.Deterministic("delta", mu[0, 1] - mu[0, 0])  # post - pre for zpdes
        # likelihood
        pm.Normal("y", mu=mu[g, t], sigma=sigma[g, t],
                  observed=data_combined[f'{task_condition}-rt'], dims="obs_id")
        # pm.LogNormal("y", mu=mu[g, t], sigma=sigma[g, t],
        #              observed=data_combined[f'{task_condition}-rt'], dims="obs_id")
    return RT_model


def build_ufov_threshold_hyper(data_combined, task_condition, coords):
    mean_prior = 60
    std_prior = 100
    with pm.Model(coords=coords) as UFOV_model:
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        # Hyperpriors for Gamma distribution parameters
        alpha_hyper = pm.Normal('alpha_hyper', mu=3.6, sigma=0.1)
        beta_hyper = pm.Normal('beta_hyper', mu=0.06, sigma=0.01)
        # Participant-level means from Gamma distribution
        mu = pm.Gamma('mu', alpha=alpha_hyper, beta=beta_hyper, dims=("condition", "time"))
        # Common standard deviation for all participants from HalfNormal
        sigma = pm.HalfNormal('sigma', std_prior, dims=("condition", "time"))
        if len(coords['condition']) > 1:
            # Deterministic parameters for the differences
            delta_zpdes = pm.Deterministic("delta_zpdes", mu[1, 1] - mu[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", mu[0, 1] - mu[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        else:
            delta = pm.Deterministic("delta", mu[0, 1] - mu[0, 0])  # post - pre for zpdes
        # likelihood with HalfNormal
        pm.Normal("y", mu=mu[g, t], sigma=sigma[g, t], observed=data_combined[f'{task_condition}-threshold'],
                  dims="obs_id")
    return UFOV_model


def build_ufov_threshold(data_combined, task_condition, coords):
    mean_prior = 60
    std_prior = 18
    with pm.Model(coords=coords) as UFOV_model:
        # Data
        g = pm.MutableData("g", data_combined.condition_idx, dims="obs_id")
        t = pm.MutableData("t", data_combined.time_idx, dims="obs_id")
        # Hyperpriors for Gamma distribution parameters
        # alpha_hyper = pm.Normal('alpha_hyper', mu=3.6, sigma=0.1)
        # beta_hyper = pm.Normal('beta_hyper', mu=0.06, sigma=0.01)
        # Participant-level means from Gamma distribution
        mu = pm.Normal('mu', mu=mean_prior, sigma=std_prior, dims=("condition", "time"))
        # Common standard deviation for all participants from HalfNormal
        sigma = pm.HalfNormal('sigma', std_prior, dims=("condition", "time"))
        if len(coords['condition']) > 1:
            # Deterministic parameters for the differences
            delta_zpdes = pm.Deterministic("delta_zpdes", mu[1, 1] - mu[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline", mu[0, 1] - mu[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        else:
            delta = pm.Deterministic("delta", mu[0, 1] - mu[0, 0])  # post - pre for zpdes
        # likelihood with HalfNormal
        pm.Normal("y", mu=mu[g, t], sigma=sigma[g, t], observed=data_combined[f'{task_condition}-threshold'],
                  dims="obs_id")
    return UFOV_model


def build_switching_cost_RT(data_combined, task_condition, coords):
    data_combined['switch_indicator'] = data_combined.apply(lambda row: literal_eval(row['switch_indicator']), axis=1)
    data_combined['results_rt'] = data_combined.apply(lambda row: literal_eval(row['results_rt']), axis=1)
    # This models needs additionnal transformation of dataset:
    coords["switch_indicator"] = ["switch", "unswitch"]
    rows = []
    for idx, row in data_combined.iterrows():
        for rt, switch in zip(row['results_rt'], row['switch_indicator']):
            rows.append({
                'participant_id': idx,
                'task_status': row['task_status'],
                'condition': row['condition'],
                'results_rt': rt,
                'switch_indicator': switch,
                'condition_idx': row['condition_idx'],
                'time_idx': row['time_idx']
            })
    df_transformed = pd.DataFrame(rows)
    # Hierarchical model
    with pm.Model(coords=coords) as switching_cost_RT:
        # Data
        g = pm.MutableData("g", df_transformed.condition_idx, dims="obs_id")
        t = pm.MutableData("t", df_transformed.time_idx, dims="obs_id")
        s = pm.MutableData("s", df_transformed.switch_indicator, dims="obs_id")
        # Hyperpriors for group means
        # mu_hyper_mean = pm.Normal('mu_hyper_mean', mu=0, sigma=10)
        # mu_hyper_sd = pm.HalfNormal('mu_hyper_sd', sigma=10)
        # Hyperpriors for group standard deviations
        # sigma_hyper_sd = pm.HalfNormal('sigma_hyper_sd', sigma=10)
        # Priors for group means given the hyperpriors
        mu = pm.Normal('mu', mu=0, sigma=10, dims=("condition", "time", "switch_indicator"))
        # Priors for group standard deviations given the hyperpriors
        sigma = pm.HalfNormal('sigma', sigma=10, dims=("condition", "time", "switch_indicator"))
        # Likelihood for each observation
        rt_obs = pm.Normal('rt_obs', mu=mu[g, t, s],
                           sigma=sigma[g, t, s], observed=df_transformed["results_rt"])
        if len(coords['condition']) > 1:
            # Deterministic node for switching cost
            switching_cost = pm.Deterministic('switching_cost', mu[:, :, 1] - mu[:, :, 0])
            # Deterministic parameters for the differences
            delta_zpdes = pm.Deterministic("delta_zpdes",
                                           switching_cost[1, 1] - switching_cost[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline",
                                              switching_cost[0, 1] - switching_cost[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        else:
            # Deterministic node for switching cost
            switching_cost = pm.Deterministic('switching_cost', mu[:, :, 1] - mu[:, :, 0])
            delta = pm.Deterministic("delta", switching_cost[0, 1] - switching_cost[0, 0])
    return switching_cost_RT


def build_switching_cost_RT_hyper(data_combined, task_condition, coords):
    data_combined['switch_indicator'] = data_combined.apply(lambda row: literal_eval(row['switch_indicator']), axis=1)
    data_combined['results_rt'] = data_combined.apply(lambda row: literal_eval(row['results_rt']), axis=1)
    # This models needs additionnal transformation of dataset:
    coords["switch_indicator"] = ["switch", "unswitch"]
    rows = []
    for idx, row in data_combined.iterrows():
        for rt, switch in zip(row['results_rt'], row['switch_indicator']):
            rows.append({
                'participant_id': idx,
                'task_status': row['task_status'],
                'condition': row['condition'],
                'results_rt': rt,
                'switch_indicator': switch,
                'condition_idx': row['condition_idx'],
                'time_idx': row['time_idx']
            })
    df_transformed = pd.DataFrame(rows)
    # Hierarchical model
    with pm.Model(coords=coords) as switching_cost_RT:
        # Data
        g = pm.MutableData("g", df_transformed.condition_idx, dims="obs_id")
        t = pm.MutableData("t", df_transformed.time_idx, dims="obs_id")
        s = pm.MutableData("s", df_transformed.switch_indicator, dims="obs_id")
        # Hyperpriors for group means
        mu_hyper_mean = pm.Normal('mu_hyper_mean', mu=0, sigma=10)
        mu_hyper_sd = pm.HalfNormal('mu_hyper_sd', sigma=10)
        # Hyperpriors for group standard deviations
        sigma_hyper_sd = pm.HalfNormal('sigma_hyper_sd', sigma=10)
        # Priors for group means given the hyperpriors
        mu = pm.Normal('mu', mu=mu_hyper_mean, sigma=mu_hyper_sd, dims=("condition", "time", "switch_indicator"))
        # Priors for group standard deviations given the hyperpriors
        sigma = pm.HalfNormal('sigma', sigma=sigma_hyper_sd, dims=("condition", "time", "switch_indicator"))
        # Likelihood for each observation
        rt_obs = pm.Normal('rt_obs', mu=mu[g, t, s],
                           sigma=sigma[g, t, s], observed=df_transformed["results_rt"])
        if len(coords['condition']) > 1:
            # Deterministic node for switching cost
            switching_cost = pm.Deterministic('switching_cost', mu[:, :, 1] - mu[:, :, 0])
            # Deterministic parameters for the differences
            delta_zpdes = pm.Deterministic("delta_zpdes",
                                           switching_cost[1, 1] - switching_cost[1, 0])  # post - pre for zpdes
            delta_baseline = pm.Deterministic("delta_baseline",
                                              switching_cost[0, 1] - switching_cost[0, 0])  # post - pre for baseline
            delta_groups = pm.Deterministic("delta_groups", delta_zpdes - delta_baseline)
        else:
            # Deterministic node for switching cost
            switching_cost = pm.Deterministic('switching_cost', mu[:, :, 1] - mu[:, :, 0])
            delta = pm.Deterministic("delta", switching_cost[0, 1] - switching_cost[0, 0])
    return switching_cost_RT


def plot_deltas(trace, ax_deltas=None, task_condition="5-nb-targets", task="moteval", y_offset=0,
                step_offset=0.2, labels=[], rope_start=-0.05, rope_end=0.05, yticks=[], model="hierarbinom",
                no_condition=False, study="v3_prolific"):
    # plt.figure(figsize=figsize)
    # plot_distribution_of_deltas(deltas, data_pre)
    vars = ["delta_zpdes", "delta_baseline"]
    condition = ["zpdes", "baseline"]
    if study == "v3_prolific":
        color_bars = [COLOR_ZPDES_v3_prolific, COLOR_BASELINE_v3_prolific]
        color_txt = ["white", "white"]
    elif study == "v3_utl":
        color_bars = [COLOR_ZPDES_v3_utl, COLOR_BASELINE_v3_utl]
        color_txt = ["white", "white"]
    elif no_condition:
        vars = ["delta"]
        condition = ["control"]
        color_bars = [COLOR_ctrl]
        color_txt = ["white"]
    else:
        color_bars = [COLOR_ZPDES_pilot, COLOR_BASELINE_pilot]
        color_txt = ["white", "white"]
    probas = [get_p_in_ROPE(trace, rope_start, rope_end, var) for var in vars]
    hdis = [get_hdi_bounds(trace, var) for var in vars]
    if model != "hierar_binom_n_var":
        labels += ["", f"{task}_{task_condition} \n {model}", ""]
    else:
        labels += ["", f"{task}_{task_condition}", ""]
    yticks += [y_offset, y_offset + step_offset / 2, y_offset + step_offset]
    for delta, p, c_b, c_t, c_l in zip(hdis, probas, color_bars, color_txt, condition):
        ax_deltas = plot_barh(delta, p, ax_deltas, height=0.5, color_bar=c_b, color_txt=c_t,
                              label=f"{task}_{task_condition}",
                              y_offset=y_offset)
        y_offset += step_offset
    return ax_deltas, y_offset, labels, yticks


def get_p_in_ROPE(trace, rope_start, rope_end, var):
    return 1 - np.mean((trace.posterior[var] > rope_start) & (trace.posterior[var] < rope_end))


def get_p_sup_ROPE(trace, rope_end, nb_samples=4000):
    return np.mean(np.mean([sum(trace[i, :].values > rope_end) for i in range(len(trace))])) / nb_samples


def get_hdi_bounds(trace, var, hdi_prob=0.94):
    # Compute the HDIs:
    vals = trace.posterior[var].values.flatten()
    # returns the hdi
    return az.hdi(vals, hdi_prob=0.94)


# Scatter plots
def jitter_yvals(y_base, val_x, existing, amount=0.05):
    """Shifts y value slightly above if it already exists in the set"""
    if val_x in existing:
        existing[val_x] += amount
    else:
        existing[val_x] = y_base
    return existing[val_x]


def plot_distribution_of_deltas(deltas, data_pre):
    # Set Y-coordinates for each condition
    y_coords = {'zpdes': 0, 'baseline': 1}
    # Overlay individual deltas
    for condition, group in deltas.groupby(data_pre['condition_idx']):
        condition_name = 'zpdes' if condition == 1 else 'baseline'
        existing_x = {}
        y_base = y_coords[condition_name]
        y_vals = [jitter_yvals(y_base, val_x, existing_x) for val_x in group]
        plt.scatter(group.values, y_vals, alpha=0.6, edgecolors='k', color='white', s=40)


def plot_barh(hdi, proba, ax, color_bar, height=0.3, label="delta_zpdes", color_txt='black', y_offset=0):
    # Plot the HDIs
    ax.barh(y_offset, width=(hdi[1] - hdi[0]), left=hdi[0], height=height, align='center', alpha=1, label=label,
            color=color_bar, edgecolor='black')
    # Displaying the probabilities in the middle of the bars
    proba_label = "" if math.isclose(proba, 0, abs_tol=0.01) else f"{proba:.1%}"
    center = hdi[0] + (hdi[1] - hdi[0]) / 2
    ax.text(center, y_offset, proba_label, ha='center', va='center', color=color_txt, fontsize=10, fontweight='bold')
    return ax


def plot_vline(rope_start, rope_end, ax):
    # Adding vertical lines
    ax.axvline(x=rope_start, color='gray', linestyle='--')
    ax.axvline(x=rope_end, color='gray', linestyle='--')


def set_ax_deltas(ax, xmin, xmax):
    # Display and set properties for x and y axes
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    # Set color for x and y axis ticks
    ax.xaxis.set_tick_params(color='black', direction='out')
    ax.yaxis.set_tick_params(color='black', direction='out')
    # Make the labels of the y-ticks smaller
    ax.tick_params(axis='y', which='both', labelsize='small')
    ax.tick_params(axis='x', which='both', labelsize='small')
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Value')
    ax.set_xlim([xmin, xmax])
    ax.set_title('94% HDI')
    ax.set_facecolor("white")
    return ax


def get_traces(studies, all_conditions):
    for study in studies:
        print("\n \n \n \n ============================================================================", study)
        for metric_type in all_conditions.keys():
            path_study = f"../outputs/{study}/results_{study}/"
            for task in all_conditions[metric_type].keys():
                path = f"{path_study}/{task}"
                # creating a new directory for outputs in case it doesn't exist:
                path_to_store = f"{path}/outputs"
                Path(path_to_store).mkdir(parents=True, exist_ok=True)
                # Open the correct dataframe
                if os.path.exists(f'{path}/{task}_lfa.csv'):
                    df = pd.read_csv(f'{path}/{task}_lfa.csv')
                    print(task, "\n =============")
                    for task_condition in all_conditions[metric_type][task]['conditions']:
                        for model in all_conditions[metric_type][task]['models']:
                            print(
                                f"\n \n \n \n ============================================================================"
                                f"\n{study}-{metric_type}-{task}-{task_condition}-{model}")
                            build_and_get_trace(df=df, task_condition=task_condition, model_type=model, nb_samples=4000,
                                                path_to_store=f"{path_to_store}/{task_condition}")


def plot_traces_and_deltas(studies, config_fig, all_conditions, plot_traces=True, plot_delta_matrices=True):
    '''
    This function allows to display 2 types of figures:
    plot_traces = True ==> plot the traces of the estimated parameter
    plot_delta_matrics = True ==> For all tasks of a study plot the delta plot (vertical bar plot)
    '''
    no_cdt_studies = ["v0_ubx", "v0_prolific"]
    for study in studies:
        print(study, "\n \n \n \n ============================================================================")
        for metric_type in all_conditions.keys():
            rope_start, rope_end = config_fig[metric_type]["rope_start"], config_fig[metric_type]["rope_end"]
            xmin, xmax = config_fig[metric_type]["xmin"], config_fig[metric_type]["xmax"]
            figsize = (config_fig[metric_type]["figsizex"], config_fig[metric_type]["figsizey"])
            fig, ax = plt.subplots(figsize=figsize, dpi=1500)
            set_ax_deltas(ax, xmin, xmax)
            path_study = f"../outputs/{study}/results_{study}/"
            y_offset, step_offset = 0, 0.5
            labels, yticks = [], []
            plot_vline(rope_start, rope_end, ax)
            for task in all_conditions[metric_type].keys():
                path = f"{path_study}/{task}"
                # creating a new directory for outputs in case it doesn't exist:
                path_to_store = f"{path}/outputs"
                Path(path_to_store).mkdir(parents=True, exist_ok=True)
                for task_condition in all_conditions[metric_type][task]['conditions']:
                    for model in all_conditions[metric_type][task]['models']:
                        var_names = config_fig[metric_type]["var_specifics"][model]
                        # Make sure that var_names are correct if no conditions:
                        control_study = study in no_cdt_studies
                        if control_study:
                            var_names = [var for var in var_names if "delta" not in var] + ["delta"]
                        if os.path.exists(f"{path_to_store}/{task_condition}-{model}_inference_data.json"):
                            # Open trace
                            trace = az.from_json(f"{path_to_store}/{task_condition}-{model}_inference_data.json")
                            # Then plot the trace:
                            if plot_traces:
                                plot_trace(trace, var_names, f"{path_to_store}/{task_condition}-{model}",
                                           model_type=model)
                            # Plot the deltas:
                            if plot_delta_matrices:
                                ax, y_offset, labels, yticks = plot_deltas(trace, ax_deltas=ax,
                                                                           task_condition=task_condition,
                                                                           task=task, y_offset=y_offset,
                                                                           step_offset=step_offset,
                                                                           labels=labels, rope_start=rope_start,
                                                                           rope_end=rope_end, yticks=yticks,
                                                                           model=model,
                                                                           no_condition=control_study, study=study)
                    # Add margin between task condition:
                    y_offset += step_offset
                # Add lines to separate between tasks:
                if plot_delta_matrices:
                    ax.hlines(y=y_offset - step_offset / 2, xmin=xmin, xmax=xmax, color="black")
                y_offset += step_offset
                labels.append("")
                yticks.append(y_offset)
            if plot_delta_matrices:
                labels, yticks = labels[:-1], yticks[:-1]
                ax.set_yticks(yticks)
                ax.set_yticklabels(labels)
                # Adjust to add grids or ticks:
                for index_lab, label in enumerate(labels):
                    if label == "":
                        ax.yaxis.get_majorticklines()[index_lab * 2].set_visible(False)
                    else:
                        ax.get_ygridlines()[index_lab].set_visible(False)
                fig.savefig(f"{path_study}/results_{metric_type}.svg", bbox_inches='tight')
                plt.close()


def get_deltas_matrices(studies, config_fig, all_conditions, add_label=""):
    '''
    From a list of stydies, this function outputs the difference between studies by using the traces
    of previously fitted models
    '''
    no_cdt_studies = ["v0_ubx", "v0_prolific"]
    for metric_type in all_conditions.keys():
        for task in all_conditions[metric_type].keys():
            for task_condition in all_conditions[metric_type][task]['conditions']:
                for model in all_conditions[metric_type][task]['models']:
                    var_names = config_fig[metric_type]["var_specifics"][model]
                    print(f"{metric_type}-{task}-{task_condition}")
                    print("===============================================================")
                    tmp_studies = [study for study in studies if
                                   os.path.exists(
                                       f"../outputs/{study}/results_{study}/{task}/outputs/{task_condition}-{model}_inference_data.json")]
                    labels = get_matrix_labels(tmp_studies, no_cdt_studies)
                    delta_matrice_mean = np.zeros((len(labels), len(labels)))
                    delta_matrice_p = np.zeros((len(labels), len(labels)))
                    # For this particular model and task_condition there is one variable of interest:
                    var = config_fig[metric_type]["var_delta_btw_study"][model]
                    rope_start, rope_end = config_fig[metric_type]["rope_start"], config_fig[metric_type]["rope_end"]
                    j = 0
                    for study_j in tmp_studies:
                        path_j = f"../outputs/{study_j}/results_{study_j}/{task}/outputs"
                        trace_j = az.from_json(f"{path_j}/{task_condition}-{model}_inference_data.json")
                        nb_row_j = 2 if study_j not in no_cdt_studies else 1
                        # reset i
                        i = 0
                        for study_i in tmp_studies:
                            nb_row_i = 2 if study_i not in no_cdt_studies else 1
                            path_i = f"../outputs/{study_i}/results_{study_i}/{task}/outputs"
                            trace_i = az.from_json(f"{path_i}/{task_condition}-{model}_inference_data.json")
                            for tmp_j in range(nb_row_j):
                                for tmp_i in range(nb_row_i):
                                    delta_tmp = trace_i.posterior[var][:, :, tmp_i, 0] - trace_j.posterior[var][:, :,
                                                                                         tmp_j, 0]
                                    delta_matrice_mean[i + tmp_i, j + tmp_j] = az.summary(delta_tmp)['mean']
                                    delta_matrice_p[i + tmp_i, j + tmp_j] = get_p_sup_ROPE(delta_tmp, rope_end)
                            # Go to next
                            i += nb_row_i
                        j += nb_row_j
                    display_matrix(labels, delta_matrice_mean, f"../outputs/matrices_mean_delta{add_label}", task,
                                   model,
                                   task_condition)
                    display_matrix(labels, delta_matrice_p, f"../outputs/matrices_p_delta{add_label}", task, model,
                                   task_condition, proba_delta=True)


def get_matrix_labels(studies, no_cdt_studies):
    labels = []
    for study in studies:
        if study in no_cdt_studies:
            labels.append(f"{study}-control")
        else:
            labels.append(f"{study}-zpdes")
            labels.append(f"{study}-baseline")
    return labels


def display_matrix(labels, matrix, path, task, model, task_condition, proba_delta=False):
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix.T, display_labels=labels)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.2f')
    if proba_delta:
        disp.im_.set_clim(0, 1)
    ax.set_xticklabels(labels, rotation=45, ha='left')
    ax.xaxis.set_tick_params(labeltop=True, labelbottom=False)
    # Add gridlines
    ax.set_xticks(np.arange(matrix.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    # Remove gridlines for major ticks
    ax.grid(False, which='major')
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.savefig(f"{path}/{task}-{task_condition}-{model}.png", bbox_inches='tight')


def generate_text(studies, config_fig, all_conditions):
    str = ""
    groups, times = ['baseline', 'zpdes'], ['pre', 'post']
    for study in studies:
        str += "V3_prolific: \n \n"
        for metric_type in all_conditions.keys():
            for task in all_conditions[metric_type].keys():
                for task_condition in all_conditions[metric_type][task]['conditions']:
                    for model in all_conditions[metric_type][task]['models']:
                        print(f"{study} - {task} - {task_condition}")
                        str += f"{metric_type} - {task} - {task_condition}: \n"
                        str += f"For the condition {task_condition}: \n"
                        path = f"../outputs/{study}/results_{study}/{task}/outputs"
                        trace = az.from_json(f"{path}/{task_condition}-{model}_inference_data.json")
                        var = config_fig[metric_type]["var_delta_btw_study"][model]
                        trace_var = trace.posterior[var]
                        trace_summary = az.summary(trace_var)
                        for time in times:
                            for grp in groups:
                                info = trace_summary.loc[f'{var}[{grp}, {time}]']
                                str += f"In the {grp} group, the average performance at {time}-test was "
                                str += f"{info['mean']}, with an HDI of "
                                str += f"[{info['hdi_3%']}," \
                                       f" {info['hdi_97%']}] (ESS_tail={info['ess_tail']}, R-hat={info['r_hat']})."
                            trace_diff = az.summary(trace_var[:, :, 1, 0] - trace_var[:, :, 0, 0])
                            str += f"The posterior difference in performance between groups at {time}-test had a mean" \
                                   f" of {trace_diff.loc[var, 'mean']}, with an HDI of [{trace_diff.loc[var, 'hdi_3%']}, " \
                                   f"{trace_diff.loc[var, 'hdi_97%']}]. \n"
                        info_zpdes = az.summary(trace.posterior["delta_zpdes"]).loc[f'delta_zpdes']
                        info_baseline = az.summary(trace.posterior["delta_baseline"]).loc[f'delta_baseline']
                        info_groups = az.summary(trace.posterior["delta_groups"]).loc[f'delta_groups']
                        str += f"\n The mean of the posterior difference between the pre-test and the post-test was " \
                               f"{info_zpdes['mean']} " \
                               f"for the ZPDES group, with an HDI of [{info_zpdes['hdi_3%']}, " \
                               f" {info_zpdes['hdi_97%']}] (ESS_tail={info_zpdes['ess_tail']}, " \
                               f"R-hat={info_zpdes['r_hat']}), and" \
                               f" {info_baseline['mean']} for the baseline " \
                               f"group with an HDI of [{info_baseline['hdi_3%']}," \
                               f" {info_baseline['hdi_97%']}](ESS_tail={info_baseline['ess_tail']}, " \
                               f"R-hat={info_baseline['r_hat']}). " \
                               f"The posterior difference in evolution " \
                               f"between the groups was {info_groups['mean']}, with an HDI " \
                               f"of [{info_groups['hdi_3%']}," \
                               f" {info_groups['hdi_97%']}] (ESS_tail={info_groups['ess_tail']}, R-hat={info_groups['r_hat']})."
                        str += "\n \n"
    print(str)


def generate_latex_table(studies, config_fig, all_conditions, time="pre", model="hierar_binom_n_var", task="moteval"):
    latex_str = "\\begin{table}[h]\n\\centering\n\\begin{adjustbox}{width=1.5\\textwidth,center=\\textwidth}\n" \
                "\\begin{tabular}{|c|c|c|M{1.8cm}|c|M{2cm}|c|c|c|}\n\\hline\n"
    latex_str += "\\textbf{Study} & \\textbf{Condition} & \\textbf{Group} & " \
                 " \\textbf{Mean Posterior} & \\textbf{HDI Range} & " \
                 "\\textbf{Difference Mean Posterior} & \\textbf{BF10} &" \
                 " \\textbf{Difference HDI Range} & \\textbf{1-P(Diff $\in$ ROPE)}" \
                 "\\\\ \\hline\n "
    groups = ['zpdes', 'baseline']
    for study in studies:
        first_study_entry = True
        for metric_type in all_conditions.keys():
            if task in all_conditions[metric_type] and model in config_fig[metric_type]["var_delta_btw_study"]:
                rope_start, rope_end = config_fig[metric_type]["rope_start"], config_fig[metric_type]["rope_end"]
                for task_condition in all_conditions[metric_type][task]['conditions']:
                    nb_row_per_study = len(all_conditions[metric_type][task]['conditions']) * len(groups)
                    path = f"../outputs/{study}/results_{study}/{task}/outputs"
                    trace = az.from_json(f"{path}/{task_condition}-{model}_inference_data.json")
                    var = config_fig[metric_type]["var_delta_btw_study"][model]
                    trace_var = trace.posterior[var]
                    trace_summary = az.summary(trace_var)
                    time_index = 0 if time == "pre" else 1
                    trace_diff = trace_var[:, :, 1, time_index] - trace_var[:, :, 0, time_index]
                    trace_summary_diff = az.summary(trace_diff)
                    info_diff = trace_summary_diff.loc[f'{var}']
                    mu, sigma = config_fig[metric_type]["var_hyp_range_sddr"][model]
                    info_diff['sddr'] = get_SDDR(trace_diff, var, {'mu': mu, 'sigma': sigma})
                    proba_diff = sum(
                        [(sum(chain.values > rope_end) + sum(chain.values < rope_start)) / (len(chain.values))
                         for chain in trace_diff]) / len(trace_diff)
                    first_task_condition_entry = True
                    for grp in groups:
                        if first_study_entry:
                            # Calculate the number of rows for this study entry
                            latex_str += f"\\multirow{{{nb_row_per_study}}}{{*}}{{{rename_str_for_latex(study)}}} & "
                            first_study_entry = False
                        else:
                            latex_str += " & "
                        if model == "switching_cost":
                            tmp_grp = 1 if grp == "zpdes" else 0
                            tmp_var = 1 if time == "post" else 0
                            info = trace_summary.loc[f"switching_cost[{tmp_grp}, {tmp_var}]"]
                        else:
                            info = trace_summary.loc[f'{var}[{grp}, {time}]']
                        if first_task_condition_entry:
                            latex_str += f" \\multirow{{{2}}}{{*}}{{{rename_str_for_latex(task_condition)}}} & " \
                                         f"{grp.capitalize()} & {info['mean']:.2f} & "
                            latex_str += f"[{info['hdi_3%']:.2f}, {info['hdi_97%']:.2f}] & "
                            # latex_str += f"{info['ess_tail']:.1f} & {info['r_hat']:.1f} & "
                            latex_str += f" \\multirow{{{2}}}{{*}}{{{info_diff['mean']:.2f}}} & "
                            latex_str += f" \\multirow{{{2}}}{{*}}{{{1 / info_diff['sddr'][0]:.2f}}} & "
                            latex_str += f" \\multirow{{{2}}}{{*}}{{[{info_diff['hdi_3%']:.2f},{info_diff['hdi_97%']:.2f}]}} & "
                            if proba_diff > 0.95:
                                latex_str += f" \\multirow{{{2}}}{{*}}{{ \\textbf{{{proba_diff:.2f}}}}} \\\\ "
                            else:
                                latex_str += f" \\multirow{{{2}}}{{*}}{{{proba_diff:.2f}}} \\\\ "
                            first_task_condition_entry = False
                        else:
                            latex_str += f" & {grp.capitalize()} & {info['mean']:.2f} & "
                            latex_str += f"[{info['hdi_3%']:.2f}, {info['hdi_97%']:.2f}] & "
                            # latex_str += f"{info['ess_tail']:.1f} & {info['r_hat']:.1f} & & &\\\\ "
                            latex_str += f" & & &\\\\ "
                        if grp == groups[-1] and task_condition == all_conditions[metric_type][task]['conditions'][-1]:
                            latex_str += "\\hline\n"
                        else:
                            if grp == groups[-1]:
                                latex_str += "\\cline{2-9}\n"
                            else:
                                latex_str += "\\cline{3-5}\n"
    latex_str += "\\end{tabular}\n\\end{adjustbox}\n\\caption{Performance of " + task + " at " + time + "-test for " + \
                 rename_str_for_latex(model) + " model.}\n" \
                                               "\\label{tab:my_label}\n\\end{table}"
    return latex_str


def get_SDDR(trace, var, parameter_priors={}):
    flat_samples = trace.values[:, 1000:].flatten()
    kde = stats.gaussian_kde(flat_samples)
    posterior_density_at_zero = kde(0)  # Evaluate the density at zero
    prior_density_at_zero = stats.norm.pdf(0, loc=parameter_priors['mu'],
                                           scale=parameter_priors['sigma'])
    sddr = posterior_density_at_zero / prior_density_at_zero
    return sddr


def generate_latex_post_table(studies, config_fig, all_conditions, time="pre", model="hierar_binom_n_var",
                              task="moteval"):
    latex_str = "\\begin{table}[h]\n\\centering\n\\begin{adjustbox}{width=1.5\\textwidth,center=\\textwidth}\n" \
                "\\begin{tabular}{|c|c|c|M{3cm}|c|c|M{3.1cm}|c|}\n\\hline\n"
    latex_str += "\\textbf{Study} & \\textbf{Condition} & \\textbf{Group} & " \
                 "\\textbf{Mean Posterior of intra-group difference} & \\textbf{1-P(Diff intra-group $\in$ ROPE)}" \
                 "& \\textbf{BF10 intra} &" \
                 "\\textbf{Mean posterior of between-group difference} & \\textbf{1-P(Diff btw-group $\in$ ROPE)} " \
                 "\\\\ \\hline\n "
    groups = ['zpdes', 'baseline']
    for study in studies:
        first_study_entry = True
        for metric_type in all_conditions.keys():
            if task in all_conditions[metric_type] and model in config_fig[metric_type]["var_delta_btw_study"]:
                rope_start, rope_end = config_fig[metric_type]["rope_start"], config_fig[metric_type]["rope_end"]
                for task_condition in all_conditions[metric_type][task]['conditions']:
                    nb_row_per_study = len(all_conditions[metric_type][task]['conditions']) * len(groups)
                    path = f"../outputs/{study}/results_{study}/{task}/outputs"
                    trace = az.from_json(f"{path}/{task_condition}-{model}_inference_data.json")
                    var = config_fig[metric_type]["var_delta_btw_study"][model]
                    trace_var = trace.posterior[var]
                    trace_summary = az.summary(trace_var)
                    time_index = 0 if time == "pre" else 1
                    # 1) Get all traces:
                    # 1.2) POST - PRE within each group:
                    evolution_trace_diff = trace_var[:, :, :, 1] - trace_var[:, :, :, 0]
                    # Check if this sig:
                    mu, sigma = config_fig[metric_type]["var_hyp_range_sddr"][model]
                    params_sddr = {'mu': mu, 'sigma': sigma}
                    sddr_z, sddr_b = get_SDDR(evolution_trace_diff[:, :, 1], var, params_sddr), get_SDDR(
                        evolution_trace_diff[:, :, 0], var, params_sddr)
                    BF = {'zpdes': 1 / sddr_z[0], 'baseline': 1 / sddr_b[0]}
                    # 1.3) Diff of diff (i.e POST-PRE diff between groups)
                    diff_diff_trace = (trace_var[:, :, 1, 1] - trace_var[:, :, 1, 0]) - (
                            trace_var[:, :, 0, 1] - trace_var[:, :, 0, 0])
                    # 2) Get all summary statistics:
                    # 2.2) POST - PRE within each group:
                    evolution_trace_summary_diff = az.summary(evolution_trace_diff)
                    # 2.3) Diff of diff (i.e POST-PRE diff between groups)
                    diff_diff_summary = az.summary(diff_diff_trace)
                    diff_diff_proba = sum(
                        [(sum(chain.values > 0)) / (len(chain.values)) for chain in diff_diff_trace]) / len(
                        diff_diff_trace)
                    first_task_condition_entry = True
                    for grp in groups:
                        grp_index = 1 if grp == "zpdes" else 0
                        if first_study_entry:
                            # Calculate the number of rows for this study entry
                            latex_str += f"\\multirow{{{nb_row_per_study}}}{{*}}{{{rename_str_for_latex(study)}}} & "
                            first_study_entry = False
                        else:
                            latex_str += " & "
                        if model == "switching_cost":
                            tmp_grp = 1 if grp == "zpdes" else 0
                            tmp_var = 1 if time == "post" else 0
                            info = trace_summary.loc[f"switching_cost[{tmp_grp}, {tmp_var}]"]
                            evolution_info_diff = evolution_trace_summary_diff.loc[f'{var}[{tmp_grp}]']
                            evolution_proba_diff = sum(
                                [(sum(chain.values > rope_end) + sum(chain.values < rope_start)) / (len(chain.values))
                                 for chain in evolution_trace_diff[:, :, grp_index]]) / len(
                                evolution_trace_diff[:, :, grp_index])
                        else:
                            info = trace_summary.loc[f'{var}[{grp}, {time}]']
                            evolution_info_diff = evolution_trace_summary_diff.loc[f'{var}[{grp}]']
                            evolution_proba_diff = sum(
                                [(sum(chain.values > rope_end) + sum(chain.values < rope_start)) / (len(chain.values))
                                 for chain in evolution_trace_diff[:, :, grp_index]]) / len(
                                evolution_trace_diff[:, :, grp_index])
                        if first_task_condition_entry:
                            # condition + group
                            latex_str += f" \\multirow{{{2}}}{{*}}{{{rename_str_for_latex(task_condition)}}} & " \
                                         f"{grp.capitalize()} &"
                            # Post - Pre for each group:
                            latex_str += f"{evolution_info_diff['mean']} & "
                            if evolution_proba_diff > 0.5:
                                latex_str += f" \\textbf{{{evolution_proba_diff:.2f}}} & "
                            else:
                                latex_str += f" {evolution_proba_diff:.2f} & "
                            latex_str += f"{BF[grp]:.2f} & "
                            # Diff evolution btw groups:
                            latex_str += f" \\multirow{{{2}}}{{*}}{{{diff_diff_summary['mean'].values[0]:.2f}}} & "
                            if diff_diff_proba > 0.5:
                                latex_str += " \\multirow{2}{*}{\\textbf{" + f"{diff_diff_proba:.2f}" + "}} \\\\ "
                            else:
                                latex_str += " \\multirow{2}{*}{" + f"{diff_diff_proba:.2f}" + "} \\\\ "
                            first_task_condition_entry = False
                        else:
                            latex_str += f" & {grp.capitalize()} & {evolution_info_diff['mean']:.2f} & "
                            if evolution_proba_diff > 0.5:
                                latex_str += f" \\textbf{{{evolution_proba_diff:.2f}}} & "
                            else:
                                latex_str += f" {evolution_proba_diff:.2f} & "
                            latex_str += f"{BF[grp]:.2f} & "
                            latex_str += f" &\\\\ "
                        if grp == groups[-1] and task_condition == all_conditions[metric_type][task]['conditions'][-1]:
                            latex_str += "\\hline\n"
                        else:
                            if grp == groups[-1]:
                                latex_str += "\\cline{2-8}\n"
                            else:
                                latex_str += "\\cline{3-6}\n"
    latex_str += "\\end{tabular}\n\\end{adjustbox}\n\\caption{Evolution of " + task + " performance for " + \
                 rename_str_for_latex(model) + " model.}\n" \
                                               "\\label{tab:my_label}\n\\end{table}"
    return latex_str


def rename_str_for_latex(input_str):
    return input_str.replace("_", "\\_")


def get_latex_tables(studies, config_fig, all_conditions):
    '''
    This function returns a file with latex tables corresponding to various results
    '''
    str = ""
    times = ["post", "pre"]
    for time in times:
        for metric_type in all_conditions.keys():
            for task in all_conditions[metric_type].keys():
                for model in all_conditions[metric_type][task]['models']:
                    if time == 'post':
                        str += generate_latex_post_table(studies, config_fig, all_conditions, time=time, model=model,
                                                         task=task)
                    str += generate_latex_table(studies, config_fig, all_conditions, time=time, model=model,
                                                task=task)
                    # After pre / post; add comparisons post - pre :
                    str += "\n \n"
                    print(f"{task}-{model}")
    text_file = open("tables_new.tex", "w")
    _ = text_file.write(str)
    text_file.close()


if __name__ == '__main__':
    '''
    The following functions allows to fit models,
     plot barplots figure for delta vars, plot delta-matrices and get latex tables of results
    '''
    # studies = ["v3_prolific", "v3_utl", "v0_ubx", "v0_prolific", "v1_prolific", "v1_ubx", "v2_prolific"]
    studies = ["v3_prolific", "v3_utl", "v0_ubx"]
    studies = ["v3_prolific"]
    studies = [ "v3_utl"]

    # with open('config/config_all-task.JSON', 'r') as f:
    with open('config/short_test.JSON', 'r') as f:
        all_conditions = json.load(f)

    with open('config/config_figures.JSON', 'r') as f:
        config_fig = json.load(f)

    COLOR_ZPDES_v3_prolific = "#114B5B"
    COLOR_BASELINE_v3_prolific = "#9CC19F"

    COLOR_ZPDES_v3_utl = "#114B5B"
    COLOR_BASELINE_v3_utl = "#9CC19F"

    COLOR_ZPDES_pilot = "#114B5B"
    COLOR_BASELINE_pilot = "#9CC19F"

    COLOR_ctrl = "#487BA7"

    get_latex_tables = False
    get_deltas_matrices = True

    if get_deltas_matrices:
        # get_traces(studies, all_conditions)
        plot_traces_and_deltas(studies, config_fig, all_conditions, plot_traces=False, plot_delta_matrices=True)
        get_deltas_matrices(studies, config_fig, all_conditions)
    
    # studies = ["v3_prolific"]
    # get_deltas_matrices(studies, config_fig, all_conditions, add_label="/restricted")
    if get_latex_tables:
        get_latex_tables(studies, config_fig, all_conditions)
