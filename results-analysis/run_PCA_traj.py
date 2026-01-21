import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from run_dr_analysis import get_scree_plot
import seaborn as sns
from matplotlib import gridspec
import imageio
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import chi2
import statsmodels.formula.api as smf
import pymc as pm
import scipy.stats as stats
import arviz as az

COLOR_ZPDES = "#114B5B"
COLOR_BASELINE = "#9CC19F"


def parse_trajectory(row):
    values = row['trajectory'][1:-1].split()
    values = np.array(list(map(lambda x: float(x), values)))
    return values


def plot_all_scatters(df, PC1, PC2, unique_ids, participant, ax):
    # Scatter plot with colors based on participant_id
    for pid in unique_ids:
        idx = participant_ids == pid
        if all(df[df['participant_id'] == pid]['condition'] == "zpdes"):
            color = COLOR_ZPDES
        else:
            color = COLOR_BASELINE
        ax.scatter(PC1[idx], PC2[idx], color=color, s=10)  # smaller markers with s=10
        if pid == participant:
            ax.plot(PC1[idx][0], PC2[idx][0], c='red', linewidth=2, marker="s", markersize=4)
    ax.set_title(f"PCA projection trajectory")


def plot_grids(grids, axes):
    # Assuming 'grids' is a list of four 3x4 numpy arrays
    cmap = plt.cm.get_cmap('gray_r', 5)
    x_labels = ['(s=low, d=high)', '(s=high, d=high)',
                '(s=low, d=low)', '(s=high, d=low)']
    y_labels = ['2-targets', '4-targets', '6-targets']
    sessions_labels = [f"Session {i}" for i in [1, 4, 5, 8]]
    for i, ax in enumerate(axes):
        im = ax.imshow(grids[i], cmap=cmap, interpolation='nearest', origin='lower', aspect='auto')
        # Set the major ticks at the centers and minor tick at the edges
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticks(np.arange(-.5, len(x_labels), 1), minor=True)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticks(np.arange(-.5, len(y_labels), 1), minor=True)
        if i > 0:  # Remove y-axis labels for all but the first subplot
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(y_labels)
        ax.set_title(sessions_labels[i])
        # Show the grid on minor ticks
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        ax.tick_params(which='minor', size=0)  # Hide the minor tick marks


def plot_mean(df, ax, participant, metric_type="binary", study="v3_prolific"):
    zpdes = df[df['condition'] == 'zpdes']
    baseline = df[df['condition'] == 'baseline']
    for index, row in df.iterrows():
        shift = 0.08 if row['condition'] == "zpdes" else -0.08
        color = COLOR_ZPDES if row['condition'] == "zpdes" else COLOR_BASELINE
        x = [i + shift for i in range(4)]
        ax.scatter(x, row[['s_0', 's_1', 's_2', 's_3']], c=color, s=1.5)
        if row['participant_id'] == participant:
            ax.plot([i for i in range(4)], row[['s_0', 's_1', 's_2', 's_3']], c='red', linewidth=2, marker="s",
                    markersize=4)
    ax.plot([i for i in range(4)], zpdes[['s_0', 's_1', 's_2', 's_3']].mean(), c=COLOR_ZPDES, marker="s",
            markersize=4, label="ZPDES")
    ax.plot([i for i in range(4)], baseline[['s_0', 's_1', 's_2', 's_3']].mean(), c=COLOR_BASELINE, marker="s",
            markersize=4, label="Baseline")
    ax.set_xticks([i for i in range(4)], ['s_1', 's_4', 's_5', 's_8'])
    ax.set_title(f"Average {metric_type} score \n {study}")


# Load data and normalize
study = "v3_utl"
metric_type = "F1"
path = f"../outputs/{study}/results_{study}/figures_latent/"
Path(path).mkdir(parents=True, exist_ok=True)
Path(f"{path}/zpdes/").mkdir(parents=True, exist_ok=True)
Path(f"{path}/baseline/").mkdir(parents=True, exist_ok=True)
df_data = pd.read_csv(f"../outputs/{study}/{study}_{metric_type}_intra.csv")
df_data['trajectory'] = df_data.apply(parse_trajectory, axis=1)
if metric_type == "F1":
    df_data['trajectory'] = df_data.apply(lambda x: np.array(x['trajectory']) / 4, axis=1)
input_data = np.vstack(df_data['trajectory'].values)
# Useful for later:
participant_ids = df_data['participant_id']
unique_ids = participant_ids.unique()

initial_size = len(input_data)
input_data = input_data.reshape((initial_size * 4, 12))
scaler = StandardScaler()
scaler.fit(input_data)
input_data = scaler.transform(input_data)

# Fit PCA and project data:
model = PCA(n_components=12)
model.fit(input_data)
get_scree_plot(model, n_components=12, path='./')
nb_components_to_keep = 2
latent_data = model.transform(input_data)
kept_latent = latent_data[:, :nb_components_to_keep]
kept_latent = kept_latent.reshape((initial_size, nb_components_to_keep * 4))
PC1, PC2 = kept_latent[:, ::2], kept_latent[:, 1::2]


def get_all_trajectories(df, PC1, PC2, unique_ids, path):
    plt.close()
    images_path = []
    # For all participants we visualize its trajectory in both latent and observed space
    for participant in unique_ids:
        condition = df[df['participant_id'] == participant]['condition'].iloc[0]
        # Plotting
        fig = plt.figure(figsize=(24, 5))
        gs = gridspec.GridSpec(2, 6, height_ratios=[1, 2], width_ratios=[1, 0.5, 0.5, 0.5, 0.5, 1])
        # Create subplots. axs[0] and axs[5] span two rows in height.
        axs = [fig.add_subplot(gs[0:2, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
               fig.add_subplot(gs[1, 3]), fig.add_subplot(gs[1, 4]), fig.add_subplot(gs[0:2, 5])]
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.4, hspace=0.4)
        # First axs[0] is the scatter plot with the trajectory of our participant
        plot_all_scatters(df, PC1, PC2, unique_ids, participant, axs[0])
        # Second axs[1:5] is the real trajectory
        grids = df[df['participant_id'] == participant]['trajectory'].values[0].reshape((4, 3, 4))
        plot_grids(grids, axs[1:5])
        # Third axs[5] is the average position on average plot
        plot_mean(df, axs[5], participant, metric_type="binary", study="v3_prolific")
        # Show plot
        fig.suptitle(f"Participant ID: {participant} \n Group: {condition}", fontsize=16)
        path_participant = f"{path}/{condition}/{participant}.png"
        images_path.append(path_participant)
        fig.savefig(path_participant)

    # Create a GIF for each participant
    with imageio.get_writer(f'{path}/results_zpdes.gif', mode='I', fps=0.5) as writer:
        for image_path in images_path:
            if "zpdes" in image_path:
                image = imageio.imread(image_path)
                writer.append_data(image)
    with imageio.get_writer(f'{path}/results_baseline.gif', mode='I', fps=0.5) as writer:
        for image_path in images_path:
            if "baseline" in image_path:
                image = imageio.imread(image_path)
                writer.append_data(image)


def get_group_trajectory(df, size, get_log=False, fit_models=True):
    nb_activity_per_evaluation = 12
    points = [[None for i in range(size)] for j in range(nb_activity_per_evaluation)]
    means, slopes = [], []
    for activity_index in range(nb_activity_per_evaluation):
        # First get all points
        for participant_idx, participant in enumerate(df['trajectory']):
            participant_traj = participant.reshape(4, 12)
            points[activity_index][participant_idx] = participant_traj[:, activity_index].tolist()
        # Then get means:
        tmp_data = np.array(points[activity_index])
        means.append(tmp_data.mean(axis=0))
        # Finally get slopes:
        n_participants, n_measurements = tmp_data.shape
        df_long = pd.DataFrame({
            'participant_id': np.repeat(np.arange(n_participants), n_measurements),
            'time': np.tile(np.arange(n_measurements), n_participants),
            'measurement': tmp_data.flatten()
        })
        if fit_models:
            # Define and fit the mixed-effects linear model with random slopes for time
            mdf_no_random_slopes = smf.mixedlm("measurement ~ time", df_long, groups=df_long["participant_id"]).fit()
            # Fit bayesian models
            if get_log:
                mdf = smf.mixedlm("measurement ~ time", df_long, groups=df_long["participant_id"],
                                  re_formula="~time").fit()
                # Fit the model without random slopes
                print(f"Activity {activity_index}")
                print(mdf_no_random_slopes.summary())
                print(mdf.summary())
            slopes.append([mdf_no_random_slopes.params['Intercept'], mdf_no_random_slopes.params['time']])
    return points, means, slopes


def plot_scatter_per_activity_on_ax(ax, idx, data, size, condition):
    c = COLOR_ZPDES if condition == "zpdes" else COLOR_BASELINE
    delta = 1 if condition == "zpdes" else -1
    shift = 0.1
    x = np.array([0, 1, 2, 3]) + delta * shift
    for participant in range(size):
        ax.scatter(x, data[idx][participant], c=c, s=0.3)


def plot_regression_line_per_activity_on_ax(ax, idx, data, condition):
    c = COLOR_ZPDES if condition == "zpdes" else COLOR_BASELINE
    # We just need 2 points to draw a line:
    x = np.array([0, 1, 2, 3])
    # Compute y based on and slope and intercept
    y = data[idx][0] + x * data[idx][1]
    ax.plot(x, y, c=c)


# Create a function to transform the data
def transform_to_long(data, group_name):
    n_participants = len(data)
    n_measurements = len(data[0])
    df = pd.DataFrame({
        'group': group_name,
        'participant_id': np.repeat(np.arange(n_participants), n_measurements),
        'time': np.tile(np.arange(n_measurements), n_participants),
        'measurement': np.array(data).flatten()
    })
    return df


def run_anonva_per_activity(data, idx):
    zpdes, baseline = data["zpdes"]["points"][idx], data["baseline"]["points"][idx]
    # Transform both datasets
    df_zpdes = transform_to_long(zpdes, 'zpdes')
    df_baseline = transform_to_long(baseline, 'baseline')
    df_combined = pd.concat([df_zpdes, df_baseline]).reset_index(drop=True)
    # Fit the model
    model = ols('measurement ~ C(group) + C(time) + C(group):C(time)', data=df_combined).fit()
    # Perform ANOVA
    anova_results = sm.stats.anova_lm(model, typ=2)
    p_val_group = anova_results.loc['C(group)', 'PR(>F)']
    p_val_time = anova_results.loc['C(time)', 'PR(>F)']
    p_val_interaction = anova_results.loc['C(group):C(time)', 'PR(>F)']
    return p_val_group, p_val_time, p_val_interaction


def study_success_in_each_cell(df, ylims=(0, 5), yticks=[0, 1, 2, 3, 4], row_vertical_val=2.5,
                               title="All participants"):
    """
    This method save the scatter plot for each activity of the intra-training and color dots with group belonging
    + compute a regression slope for each group and plot the corresponding line
    + Test for each activity whether significant interactions
    """
    save_string = ""
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
    fig.subplots_adjust(wspace=0, hspace=0)
    conditions = {"zpdes": {"points": [], "means": [], "slopes": []},
                  "baseline": {"points": [], "means": [], "slopes": []}}
    for condition, condition_dict in conditions.items():
        # First restrict to only a specific condition:
        tmp_data = df[df['condition'] == condition]
        tmp_size = len(tmp_data)
        # Then get an array of size 48 where each element is of size 4 x len(group)
        condition_dict["points"], condition_dict["means"], condition_dict["slopes"] = get_group_trajectory(tmp_data,
                                                                                                           tmp_size)
        # After getting the data just plot it on the ax:
        for idx, ax in enumerate(axes.flatten()):
            plot_scatter_per_activity_on_ax(ax, idx, condition_dict["points"], tmp_size, condition)
            plot_regression_line_per_activity_on_ax(ax, idx, condition_dict["slopes"], condition)
    for idx, ax in enumerate(axes.flatten()):
        # Use this loop to set the axis in correct display mode:
        ax.set_xticks([0, 1, 2, 3], ["S1", "S4", "S5", "S9"])
        ax.set_ylim(ylims)
        ax.set_yticks(yticks)
        if idx % 4 != 0:
            ax.set_yticks([])
        if idx in [1, 2, 3, 5, 6, 7]:
            ax.set_xticks([])
        cols = ["speed=easy, dist=easy", "speed=hard, dist=easy", "speed=easy, dist=hard", "speed=hard, dist=hard", ]
        if idx in [0, 1, 2, 3]:
            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()
            ax.set_xticks([1.5])  # Assuming 5 is the center for the x-axis
            ax.set_xticklabels([cols[idx]], va='bottom')
        rows = ["nb_targets=easy", "nb_targets=medium", "nb_targets=hard"]
        if idx in [3, 7, 11]:
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()
            ax.set_yticks([row_vertical_val])  # Assuming 5 is the center for the x-axis
            ax.set_yticklabels([rows[idx % 3]], va='center')
        # Fit an ANOVA to look for sig difference
        # If sig diff, add on ax:
        p_val_group, p_val_time, p_val_interaction = run_anonva_per_activity(conditions, idx)
        dict, tmp_str = get_bayesian_mixed_lm_model(conditions, idx)
        save_string += f"Activity index: {idx} \n" + tmp_str + "\n"
        # 'group_slope', 'session_id_slope', 'group_session_interaction'
        # ax.text(0.5, 0.95,
        #         f'Grp HDI=[{dict["group_slope"]["3%"]:.2f}, {dict["group_slope"]["97%"]:.2f}]- BF={dict["group_slope"]["BF"][0]:.2f}, \n'
        #         f'Time HDI=[{dict["session_id_slope"]["3%"]:.2f}, {dict["session_id_slope"]["97%"]:.2f}] - BF={dict["session_id_slope"]["BF"][0]:.2f} \n'
        #         f'Inter HDI=[{dict["group_session_interaction"]["3%"]:.2f}, {dict["group_session_interaction"]["97%"]:.2f}] - BF={dict["group_session_interaction"]["BF"][0]:.2f}',
        #         ha='center', va='top',
        #         transform=ax.transAxes)
    fig.suptitle(
        title + f" (ZPDES={len(df[df['condition'] == 'zpdes'])}, Baseline={len(df[df['condition'] == 'baseline'])})")
    plt.savefig(f"{title}-in-cell.png")
    with open(f"{title}_data.txt", 'w') as file:
        file.write(save_string)


def get_bayesian_mixed_lm_model(data, idx, range_max=1):
    # Converting categorical variables to codes
    df_zpdes, df_baseline = pd.DataFrame(data['zpdes']['points'][idx]), pd.DataFrame(data['baseline']['points'][idx])
    df_zpdes['condition'], df_baseline['condition'] = 'zpdes', 'baseline'
    df = pd.concat([df_zpdes, df_baseline], ignore_index=True)
    df['participant_id'] = range(1, len(df) + 1)
    data = pd.melt(df, id_vars=['participant_id', 'condition'], value_vars=[i for i in range(4)],
                   var_name='timepoint', value_name='measurement')
    data['group_code'] = data['condition'].astype('category').cat.codes
    data['id_code'] = data['participant_id'].astype('category').cat.codes
    n_groups = len(data['participant_id'].unique())
    with pm.Model() as model:
        id_code = pm.Data('id_code', data['id_code'])
        timepoint = pm.Data('timepoint', data['timepoint'])
        # Priors
        intercept = pm.Normal('Intercept', mu=range_max, sigma=range_max / 10)
        group_slope = pm.Normal('group_slope', mu=0, sigma=0.05)
        session_id_slope = pm.Normal('session_id_slope', mu=0, sigma=0.05)
        group_session_interaction = pm.Normal('group_session_interaction', mu=0, sigma=0.05)
        sigma = pm.HalfNormal('sigma', sigma=0.1)
        participant_intercept = pm.Normal('participant_Intercept', mu=0, sigma=range_max / 10, shape=n_groups)
        # Expected value
        mu = (intercept +
              group_slope * data['group_code'] +
              session_id_slope * timepoint +
              group_session_interaction * data['group_code'] * timepoint +
              participant_intercept[id_code])
        # Likelihood
        likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=data['measurement'])
        # Inference
        trace = pm.sample(5000, tune=1000, target_accept=0.95)
    # Summary
    summary_df = az.summary(trace,
                            var_names=[var for var in trace.posterior.data_vars if var != 'participant_Intercept'])
    # Probability calculations
    rope = 0.001
    # A dictionary to hold the SDDR values
    sddr_values = {}
    # Define the parameters based on their priors in the model
    parameter_priors = {
        'group_slope': {'mu': 0, 'sigma': 0.05},
        'session_id_slope': {'mu': 0, 'sigma': 0.05},
        'group_session_interaction': {'mu': 0, 'sigma': 0.05},
    }
    summary_str = "Bayesian model summary:\n"
    dict = {}
    for varname in ['Intercept', 'group_slope', 'session_id_slope', 'group_session_interaction']:
        dict[varname] = {}
        flat_samples = trace.posterior[varname].values[:, 1000:].flatten()
        p_positive = (flat_samples > 0).mean()
        p_threshold = (abs(flat_samples) > rope).mean()
        summary_df.loc[varname, 'P(>0)'] = p_positive
        summary_df.loc[varname, f'P(||>{rope})'] = p_threshold
        direction = 'positive' if np.median(flat_samples) > 0 else 'negative'
        prob_in_direction = p_positive if direction == "positive" else 1 - p_positive
        dict[varname]['p_effect'] = prob_in_direction
        dict[varname]['3%'] = summary_df.loc[varname, 'hdi_3%']
        dict[varname]['97%'] = summary_df.loc[varname, 'hdi_97%']
        summary_df.loc[varname, f'P(dir>{rope})'] = prob_in_direction
        # SDDR:
        if varname in parameter_priors:
            kde = stats.gaussian_kde(flat_samples)
            posterior_density_at_zero = kde(0)  # Evaluate the density at zero
            prior_density_at_zero = stats.norm.pdf(0, loc=parameter_priors[varname]['mu'],
                                                   scale=parameter_priors[varname]['sigma'])
            sddr = posterior_density_at_zero / prior_density_at_zero
            summary_df.loc[varname, 'SDDR'] = sddr
            dict[varname]['BF'] = sddr
    summary_str += summary_df.to_string()
    return dict, summary_str


if __name__ == '__main__':
    if metric_type == "F1":
        yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        row_vertical_val = 0.6
        ylims = (0, 1.2)
    else:
        yticks = [0, 1, 2, 3, 4]
        row_vertical_val = 2.5
        ylims = (0, 5)
    # Calculate the median score
    median_score = df_data['s_0'].median()
    # Calculate the third and sixth deciles
    third_decile = df_data['s_0'].quantile(0.3)
    sixth_decile = df_data['s_0'].quantile(0.6)
    # Split the DataFrame into two based on the median score
    low_performer = df_data[df_data['s_0'] <= third_decile]
    medium_performer = df_data[(df_data['s_0'] > third_decile) & (df_data['s_0'] <= sixth_decile)]
    high_performer = df_data[df_data['s_0'] > sixth_decile]
    study_success_in_each_cell(df_data, title="All participants", ylims=ylims, yticks=yticks,
                               row_vertical_val=row_vertical_val)
    # study_success_in_each_cell(low_performer, title="Low performers", ylims=ylims, yticks=yticks,
    #                            row_vertical_val=row_vertical_val)
    # study_success_in_each_cell(medium_performer, title="Medium performers", ylims=ylims, yticks=yticks,
    #                            row_vertical_val=row_vertical_val)
    # study_success_in_each_cell(high_performer, title="High performers", ylims=ylims, yticks=yticks,
    #                            row_vertical_val=row_vertical_val)
    get_all_trajectories(df_data, PC1, PC2, unique_ids, path)
