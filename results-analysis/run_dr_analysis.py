import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from pathlib import Path
import pickle
from scipy.stats import norm
from sklearn.model_selection import KFold
from scipy.stats import ttest_rel, ttest_ind, rankdata
import seaborn as sns
from pingouin import ttest


# Display the PCA:
def get_scatter_of_indiv_2D(df_pca, path):
    # Create a scatter plot of the first two principal components
    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca['PC1'], df_pca['PC2'], marker='o', c='b', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA - Scatter Plot of Principal Components 1 and 2')
    plt.grid()
    # Show or save the plot
    plt.savefig(f"{path}/scatter_plots.png")


def create_correlation_circle(variable_contributions, variable_names, path):
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 8))
    # Define a colormap for arrows
    cmap = plt.get_cmap('tab10')
    # Plot the correlation circle (arrows) with unique colors and labels
    for i, var_name in enumerate(variable_names):
        arrow_length = np.sqrt(variable_contributions[i, 0] ** 2 + variable_contributions[i, 1] ** 2)
        arrow_x = variable_contributions[i, 0] / arrow_length
        arrow_y = variable_contributions[i, 1] / arrow_length
        color = cmap(i % 10)  # Use modulo to cycle through colors
        plt.arrow(0, 0, arrow_x, arrow_y, color=color, alpha=0.7, label=var_name)
    # Add a circle of radius 1
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='dotted', alpha=0.5)
    ax.add_artist(circle)
    # Add a legend on the right of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    # Set axis limits to show the circle
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    # Show or save the plot
    plt.savefig(f"{path}/correlation_circle.png")


def get_heat_and_dendro_contrib(variable_contributions, variable_names, n_components, path):
    # Assuming you have variable_contributions, a 2D numpy array with the variable contributions to components
    # Also, make sure you have the original variable names in a list, e.g., variable_names
    # Keep only the first 10 components
    # variable_contributions = variable_contributions[:, :10]
    # Create a linkage matrix for hierarchical clustering
    linkage_matrix = linkage(variable_contributions, method='single')
    # Perform hierarchical clustering and get the reordered indices
    order = leaves_list(linkage_matrix)
    # Reorder the variable names based on clustering
    ordered_variable_names = [variable_names[i] for i in order]
    # Create a larger figure for the heatmap
    fig, ax1 = plt.subplots(figsize=(12, 8))
    # Create a heatmap of the reordered variable contributions with a custom colormap
    cmap = LinearSegmentedColormap.from_list("custom_colormap", [(0, 'blue'), (0.5, 'white'), (1, 'red')])
    cax = ax1.matshow(variable_contributions[order, :], cmap=cmap, aspect='auto', vmin=-1, vmax=1,
                      extent=(-0.5, n_components - 0.5, len(variable_names) - 0.5, -0.5))
    # Invert the y-axis to match the dendrogram
    ax1.invert_yaxis()
    ax1.set_yticks(np.arange(len(ordered_variable_names)))
    ax1.set_yticklabels(ordered_variable_names)
    ax1.set_xticks(np.arange(n_components))
    ax1.set_xticklabels([f'PC{i + 1}' for i in range(n_components)], rotation=0)
    # Add text labels with values in black
    for i in range(len(variable_names)):
        for j in range(n_components):
            value = variable_contributions[order, :][i, j]
            ax1.text(j, i, f'{value:.2f}', ha='center', va='center', color='black', fontsize=10, fontweight='bold')
    # Add a color bar for reference
    cbar = fig.colorbar(cax, ax=ax1, ticks=[-1, 0, 1])
    cbar.set_label('Loading')
    plt.title('Variable Loadings for Principal Components (First 10 Components) - Heatmap')
    fig.savefig(f"{path}/heatmap.png")
    plt.close()
    # Create a separate figure for the dendrogram
    fig2, ax2 = plt.subplots(figsize=(3, 8))
    # Create a dendrogram for hierarchical clustering with custom labels
    dendrogram(linkage_matrix,
               ax=ax2,
               orientation='left',
               # color_threshold=0.5 * np.max(linkage_matrix[:, 2]),
               distance_sort='descending',
               show_leaf_counts=True,
               leaf_font_size=20,
               color_threshold=0
               )
    ax2.set_yticklabels(ordered_variable_names)
    plt.title('Hierarchical Clustering Dendrogram')
    fig2.set_size_inches(10, 8)
    fig2.tight_layout()
    fig2.savefig(f"{path}/dendro.png")


def get_scree_plot(pca_object, n_components, path):
    # Generate Scree Plot
    n_components_kaiser = sum(pca_object.explained_variance_ > 1)
    variance_explained_kaiser = sum(pca_object.explained_variance_ratio_[:n_components_kaiser]) * 100
    plt.figure(figsize=(10, 6))
    # Create bar plot for explained variance
    plt.bar(range(1, n_components + 1), pca_object.explained_variance_ratio_, alpha=0.7, color='blue',
            label='Variance Explained by Each PC')
    # Add line plot on top of the bar plot
    plt.plot(range(1, n_components + 1), pca_object.explained_variance_ratio_, marker='o', linestyle='--', color='red',
             label='Cumulative Variance Explained')
    # Add a vertical line for the Kaiser Criterion
    plt.vlines(x=n_components_kaiser, ymin=0, ymax=1.1 * pca_object.explained_variance_ratio_[0], colors='green',
               linestyles='dashed', label='Kaiser Criterion')
    plt.title(f'Scree Plot - \n Variance explained by kept components: {variance_explained_kaiser:.2f} ')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.legend()
    plt.savefig(f"{path}/scree_plots.png")


def plot_correlation_matrix(df_transformed, path, name="main"):
    correlation_matrix = df_transformed.loc[:, 'PC1':'PC6'].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation Matrix')
    plt.savefig(f"{path}/{name}_model_correlations.png")


def plot_loading_matrix(loadings_df, variable_names, n_components, path, name):
    # Assuming you have variable_contributions, a 2D numpy array with the variable contributions to components
    # Also, make sure you have the original variable names in a list, e.g., variable_names
    # Keep only the first 10 components
    # variable_contributions = variable_contributions[:, :10]
    # Create a linkage matrix for hierarchical clustering
    loadings = loadings_df.values
    linkage_matrix = linkage(loadings, method='single')
    # Perform hierarchical clustering and get the reordered indices
    order = leaves_list(linkage_matrix)
    # order = [9, 8, 21, 19, 15, 18, 16, 17, 20, 22, 4, 3, 2, 7, 6, 5, 10, 12, 13, 14, 11, 0, 1, 23]
    order = [i for i in range(24)]
    order.reverse()
    # Reorder the variable names based on clustering
    ordered_variable_names = [variable_names[i] for i in order]
    # Create a larger figure for the heatmap
    fig, ax1 = plt.subplots(figsize=(12, 8))
    # Create a heatmap of the reordered variable contributions with a custom colormap
    # cmap = LinearSegmentedColormap.from_list("custom_colormap", [(0, 'blue'), (0.5, 'white'), (1, 'red')])
    cax = ax1.matshow(loadings[order, :n_components], cmap="PuOr_r", aspect='auto',
                      extent=(-0.5, n_components - 0.5, len(variable_names) - 0.5, -0.5))
    # Invert the y-axis to match the dendrogram
    ax1.invert_yaxis()
    ax1.set_yticks(np.arange(len(ordered_variable_names)))
    ax1.set_yticklabels(ordered_variable_names)
    ax1.set_xticks(np.arange(n_components))
    ax1.set_xticklabels([f'PC{i + 1}' for i in range(n_components)], rotation=0)
    # Add text labels with values in black
    for i in range(len(variable_names)):
        for j in range(n_components):
            value = loadings[order, :][i, j]
            ax1.text(j, i, f'{value:.2f}', ha='center', va='center', color='black', fontsize=10, fontweight='bold')
    # Add a color bar for reference
    cbar = fig.colorbar(cax, ax=ax1, ticks=[-1, 0, 1])
    cbar.set_label('Loading')
    plt.title(f'Variable Loadings for Principal Components (First {n_components} Components) - Heatmap')
    fig.tight_layout()
    fig.savefig(f"{path}/heatmap_{name}.png")
    plt.close()


# Then Functions to run the dim reduction:
def fill_na_with_mean(data, columns_for_pca):
    for study in data['study'].unique():
        study_mean = data.loc[data['study'] == study, columns_for_pca].mean()
        data.loc[data['study'] == study, columns_for_pca] = data.loc[data['study'] == study, columns_for_pca].fillna(
            study_mean)
    return data


def zeros_ones_transform(row, n=10):
    if row == 0:
        return 1 / (2 * n)
    elif row == 1:
        return 1 - (1 / (2 * n))
    return row


def normalize_data(data, columns_for_pca, columns_nb_trials, mode="zscore", shuffle=True):
    if mode == "zscore":
        # Create a StandardScaler instance
        scaler = StandardScaler()
        # Fit the scaler on the selected columns and transform the data
        df_normalized = scaler.fit_transform(data[columns_for_pca])
        # Create a new DataFrame with the scaled data
        df_normalized = pd.DataFrame(df_normalized, columns=columns_for_pca)
    elif mode == "probit":
        # First transform 0-1 values
        # Then probit transform
        df_normalized = copy.deepcopy(data)
        for index_col, col in enumerate(columns_for_pca):
            if "accuracy" in col:
                df_normalized[col] = [zeros_ones_transform(data_val, nb_val) for nb_val, (counter, data_val) in
                                      zip(df_normalized[columns_nb_trials[index_col]], enumerate(df_normalized[col]))]
                try:
                    df_normalized[col] = df_normalized[col].apply(norm.ppf)
                except:
                    print("af")
            elif "rt" in col:
                scaler = StandardScaler()
                # Fit the scaler on the selected columns and transform the data
                # df_normalized[col] = scaler.fit_transform(df_normalized[[col]])
        # Center values (again???):
        df_normalized[columns_for_pca] = (df_normalized[columns_for_pca] - df_normalized[columns_for_pca].mean()) / \
                                         df_normalized[columns_for_pca].std()
        df_normalized = df_normalized[columns_for_pca]
    elif mode == "percentile_rank":
        df_normalized = pd.DataFrame(columns=columns_for_pca)
        for col in columns_for_pca:
            df_normalized[col] = rankdata(data[col], method='min') / len(data[col]) * 100
            df_normalized[columns_for_pca] = (df_normalized[columns_for_pca] - df_normalized[columns_for_pca].mean()) / \
                                             df_normalized[columns_for_pca].std()
    else:
        df_normalized = None
        print("Non existing normalization")
    if shuffle:
        df_normalized = df_normalized.sample(frac=1, random_state=0)
    return df_normalized


def validate_model(dataframe, test, model):
    # model is trained one using pca.fit or ica.fit
    data_extract = copy.deepcopy(dataframe.values[test, :])
    codes = model.transform(data_extract)
    # test prediction error:
    data_est = model.inverse_transform(codes)
    rmse = np.sqrt(np.mean((data_extract.flatten() - data_est.flatten()) ** 2))
    correlation = np.corrcoef(data_extract.flatten(), data_est.flatten())
    correlation = correlation[0, 1]
    return rmse, correlation


# Assuming you have df_normalized, the DataFrame with centered and scaled data
def get_DR(DR_type, df_normalized, columns_for_pca, n_components=5, n_splits=10):
    # We are going to run K-fold cross validation:
    kf = KFold(n_splits=n_splits)
    models, rmses, explained_vars, correlations, components = [], [], [], [], []
    for train, test in kf.split(df_normalized):
        sub_train_df = copy.deepcopy(df_normalized.values[train, :])
        # Fit the PCA model to the data
        DR_algo = model_choice(DR_type, df_normalized, n_components)
        DR_algo.fit(sub_train_df)
        rmse, correlation = validate_model(df_normalized, test, DR_algo)
        models.append(DR_algo)
        rmses.append(rmse)
        correlations.append(correlation)
        components.append(DR_algo.components_)
        if DR_type == "PCA":
            explained_vars.append((np.cumsum(DR_algo.explained_variance_ratio_)[n_components - 1]))
    # Average of all models - not recommended but just to make sure we fit with previous work:
    mean_components = np.mean(components, 0)
    average_loadings = pd.DataFrame(np.reshape(mean_components, [n_components, df_normalized.shape[1]]),
                                    columns=df_normalized.columns).T
    # Transform the data to the principal components
    main_model = model_choice(DR_type, df_normalized, n_components)
    main_model.fit(df_normalized)
    main_model_loadings = pd.DataFrame(np.reshape(main_model.components_, [n_components, df_normalized.shape[1]]),
                                       columns=df_normalized.columns).T
    latent_df = main_model.transform(df_normalized)
    # Create a new DataFrame with the principal components
    latent_df = pd.DataFrame(data=latent_df, columns=[f'PC{i + 1}' for i in range(n_components)])
    # Assuming you have df_normalized, the DataFrame with the centered and scaled data,
    # and a list of variable names, calculate the variable contributions (correlations with principal components)
    main_variable_contributions = np.corrcoef(df_normalized.T, latent_df.T)[:len(columns_for_pca), -n_components:]
    return latent_df, average_loadings, main_model_loadings, main_variable_contributions, main_model, rmses, correlations


def model_choice(DR_type, df, n_components):
    if DR_type == "PCA":
        model = PCA(n_components=n_components)
    elif DR_type == "ICA":
        model = FastICA(n_components=n_components, random_state=0, whiten='unit-variance')
    else:
        model = None
        print("Model doesn't exit")
    return model


def run_dimensionality_reduction(data, columns_for_pca, columns_nb_trials, n_components=5, DR_type='PCA',
                                 normalization_mode="probit"):
    filled_data = fill_na_with_mean(copy.deepcopy(data), columns_for_pca)
    df_normalized = normalize_data(filled_data, columns_for_pca, columns_nb_trials,
                                   mode=normalization_mode)
    return get_DR(DR_type, df_normalized, columns_for_pca, n_components)


def save_results(df_pca, n_components, DR_type, columns_for_pca, variable_contributions, model, path, average_loadings,
                 main_model_loadings, rmses, correlations):
    # Save loadings + errors
    average_loadings.to_csv(f"{path}/average_loadings.csv")
    main_model_loadings.to_csv(f"{path}/kept_main_loadings.csv")
    metrics = pd.DataFrame.from_dict({"mean_mean": [np.mean(rmses)],
                                      "std_rmse": [np.std(rmses)],
                                      "standard_error_rmse": [np.std(rmses) / np.sqrt(len(rmses))],
                                      "mean_correl": [np.mean(correlations)],
                                      "error_correl": [np.std(correlations) / np.sqrt(len(correlations))]
                                      })
    metrics.to_csv(f"{path}/cross_validation_metrics.csv")
    # First save the model PCA object to re-use it later:
    with open(f'{path}/model.pkl', 'wb') as file:
        pickle.dump(model, file)
    get_scatter_of_indiv_2D(df_pca, path)
    if DR_type == "PCA":
        get_scree_plot(model, n_components, path)
    create_correlation_circle(variable_contributions, columns_for_pca, path)
    get_heat_and_dendro_contrib(variable_contributions, columns_for_pca, n_components, path)
    n_components_to_keep = 6
    plot_loading_matrix(average_loadings, columns_for_pca, n_components_to_keep, path, name="average")
    plot_loading_matrix(main_model_loadings, columns_for_pca, n_components_to_keep, path, name="main")
    plot_correlation_matrix(df_pca, path, name="main")


def get_df(studies):
    df = pd.DataFrame()
    for study in studies:
        tmp = pd.read_csv(f"../outputs/{study}/all_cognitive_results_{study}.csv")
        tmp['study'] = study
        df = pd.concat([df, tmp], ignore_index=True)
    return df


def get_diff_from_PCA(df_analysis, main_model, tasks, tasks_nb, study, path):
    # Use the found transformation
    df_analysis[tasks] = df_analysis[tasks].fillna(df_analysis[tasks].mean())
    cols = [col for col in df_analysis.columns if col not in kept_columns]
    df_analysis[tasks] = normalize_data(df_analysis[cols], tasks, tasks_nb, mode="probit", shuffle=False)
    df_analysis.drop(columns=tasks_nb, inplace=True)
    compute_and_plot_diff(df_analysis, main_model, tasks, study, path)


def compute_and_plot_diff(df_analysis, main_model, tasks, study, path, name="main"):
    pre, post = df_analysis[df_analysis['task_status'] == 'PRE_TEST'], df_analysis[
        df_analysis['task_status'] == 'POST_TEST']
    pre_zpdes, pre_baseline = pre[pre['condition'] == "zpdes"], pre[pre['condition'] == "baseline"]
    post_zpdes, post_baseline = post[post['condition'] == "zpdes"], post[post['condition'] == "baseline"]
    # Now project the data:
    pre_zpdes[tasks], post_zpdes[tasks] = main_model.transform(pre_zpdes[tasks]), main_model.transform(
        post_zpdes[tasks])
    pre_baseline[tasks], post_baseline[tasks] = main_model.transform(pre_baseline[tasks]), main_model.transform(
        post_baseline[tasks])
    PC_names = {col_task: f"PC{i}" for i, col_task in enumerate(tasks)}
    pre_zpdes.rename(columns=PC_names, inplace=True)
    post_zpdes.rename(columns=PC_names, inplace=True)
    pre_baseline.rename(columns=PC_names, inplace=True)
    post_baseline.rename(columns=PC_names, inplace=True)
    Path(f"{path}/ttests_data/").mkdir(parents=True, exist_ok=True)
    for pc_name in PC_names.values():
        plot_diff_on_PC(pre_baseline, post_baseline, pre_zpdes, post_zpdes, pc_name, study, path, name=name)


def masked_analysis(df_analysis, main_model, tasks, tasks_nb, study, path):
    # Data normalization
    df_analysis[tasks] = df_analysis[tasks].fillna(df_analysis[tasks].mean())
    cols = [col for col in df_analysis.columns if col not in kept_columns]
    df_analysis[tasks] = normalize_data(df_analysis[cols], tasks, tasks_nb, mode="probit", shuffle=False)
    df_analysis.drop(columns=tasks_nb, inplace=True)
    # Model masking:
    for pc_index, (pc, features) in enumerate(masks[expe_name].items()):
        kept_features = [df[tasks].columns.get_loc(feature) for feature in features]
        masked_features = [feature for feature in range(len(df[tasks].columns)) if feature not in kept_features]
        main_model.components_[pc_index, masked_features] = 0
    # Loadings to plot (verification mainly):
    main_model_loadings = pd.DataFrame(
        np.reshape(main_model.components_, [n_components, df_analysis[tasks].shape[1]]),
        columns=df_analysis[tasks].columns).T
    plot_loading_matrix(main_model_loadings, tasks, 6, path, name=f"masked_main_{study}")
    df_analysis[tasks] = main_model.transform(df_analysis[tasks])
    PC_names = {col_task: f"PC{i}" for i, col_task in enumerate(tasks)}
    kept_df = copy.deepcopy(df_analysis[tasks])
    kept_df.rename(columns=PC_names, inplace=True)
    plot_correlation_matrix(kept_df, path, name="masked")
    compute_and_plot_diff(df_analysis, main_model, tasks, study, path, name="masked")


def plot_diff_on_PC(pre_baseline, post_baseline, pre_zpdes, post_zpdes, PC_name, study, path, name="main"):
    # Merge the two dataframes
    plt.close()
    fig, ax = plt.subplots()
    # t_statistic_baseline, p_value_baseline = ttest_rel(pre_baseline[PC_name], post_baseline[PC_name])
    # t_statistic_zpdes, p_value_zpdes = ttest_rel(pre_zpdes[PC_name], post_zpdes[PC_name])
    # t_statistic_pre, p_value_pre = ttest_ind(pre_zpdes[PC_name], pre_baseline[PC_name])
    # t_statistic_post, p_value_post = ttest_ind(post_zpdes[PC_name], post_baseline[PC_name])
    diff_baseline = ttest(pre_baseline[PC_name], post_baseline[PC_name], paired=True)
    diff_zpdes = ttest(pre_zpdes[PC_name], post_zpdes[PC_name], paired=True)
    diff_pre = ttest(pre_zpdes[PC_name], pre_baseline[PC_name], paired=False)
    diff_post = ttest(post_zpdes[PC_name], post_baseline[PC_name], paired=False)
    save = pd.DataFrame.from_dict({"t_diff_zpdes": [diff_zpdes["T"].values[0]],
                                   "p_diff_zpdes": [diff_zpdes["p-val"].values[0]],
                                   "d_diff_zpdes": [diff_zpdes["cohen-d"].values[0]],
                                   "BF_diff_zpdes": [diff_zpdes["BF10"].values[0]],
                                   "t_diff_baseline": [diff_baseline["T"].values[0]],
                                   "p_diff_baseline": [diff_baseline["p-val"].values[0]],
                                   "d_diff_baseline": [diff_baseline["cohen-d"].values[0]],
                                   "BF_diff_baseline": [diff_baseline["BF10"].values[0]],
                                   "t_diff_pre": [diff_pre["T"].values[0]],
                                   "p_diff_pre": [diff_pre["p-val"].values[0]],
                                   "d_diff_pre": [diff_pre["cohen-d"].values[0]],
                                   "BF_diff_pre": [diff_pre["BF10"].values[0]],
                                   "t_diff_post": [diff_post["T"].values[0]],
                                   "p_diff_post": [diff_post["p-val"].values[0]],
                                   "d_diff_post": [diff_post["cohen-d"].values[0]],
                                   "BF_diff_post": [diff_post["BF10"].values[0]],
                                   }
                                  )
    save.to_csv(f"{path}/ttests_data/{PC_name}-{study}-{name}.csv")
    plt.scatter(pre_baseline[PC_name], post_baseline[PC_name], alpha=0.7, c='blue', label="baseline")
    plt.scatter(pre_zpdes[PC_name], post_zpdes[PC_name], alpha=0.7, c='red', label="zpdes")
    plt.plot([i for i in range(-6, 6)], [i for i in range(-6, 6)])
    plt.xlabel('Pre')
    plt.ylabel('Post')
    plt.title(f'{PC_name}')
    plt.grid(True)
    plt.legend()
    fig.savefig(f"{path}/{PC_name}-{study}-{name}-diffplot.png")


def compute_broad_aggregate_difference(df, path_to_store, tasks, study, normalization_type="zscore"):
    # 1) Easy aggregation
    # First for RT values, just take the inverse:
    rt_cols = [col for col in df.columns if "rt" in col and "cost" not in col and "participant" not in col]
    df[rt_cols] = df[rt_cols].replace(0, np.nan).apply(lambda x: 1 / x)
    # Then normalize the data using z-scores
    df[tasks] = df[tasks].fillna(df[tasks].mean())
    df[tasks] = normalize_data(df, tasks, tasks, mode=normalization_type, shuffle=False)
    save_aggregate(df, tasks, path_to_store, normalization_type, study)


def save_aggregate(df, tasks, path_to_store, normalization_type, study):
    # Split into pre-post
    pre, post = df[df['task_status'] == "PRE_TEST"], df[df['task_status'] == "POST_TEST"]
    # Split into baseline-zpdes
    zpdes_pre, zpdes_post = pre[pre['condition'] == "zpdes"][tasks], post[post['condition'] == "zpdes"][tasks]
    baseline_pre, baseline_post = pre[pre['condition'] == "baseline"][tasks], post[post['condition'] == "baseline"][
        tasks]
    # Compute the mean z-score for each participant
    mean_pre_zpdes, mean_post_zpdes = zpdes_pre.mean(axis=1).astype(float), zpdes_post.mean(axis=1).astype(float)
    mean_pre_baseline, mean_post_baseline = baseline_pre.mean(axis=1).astype(float), baseline_post.mean(axis=1).astype(float)
    # Compute ttest
    # t_diff_zpdes, p_diff_zpdes = ttest_rel(mean_pre_zpdes, mean_post_zpdes)
    # t_diff_baseline, p_diff_baseline = ttest_rel(mean_pre_baseline, mean_post_baseline)
    # t_diff_pre, p_diff_pre = ttest_ind(pd.to_numeric(mean_pre_zpdes, errors='coerce'),
    #                                    pd.to_numeric(mean_pre_baseline, errors='coerce'))
    # t_diff_post, p_diff_post = ttest_ind(pd.to_numeric(mean_post_zpdes, errors='coerce'),
    #                                      pd.to_numeric(mean_post_baseline, errors='coerce'))
    # save = pd.DataFrame.from_dict({"t_diff_zpdes": [t_diff_zpdes],
    #                                "p_diff_zpdes": [p_diff_zpdes],
    #                                "t_diff_baseline": [t_diff_baseline],
    #                                "p_diff_baseline": [p_diff_baseline],
    #                                "t_diff_pre": [t_diff_pre],
    #                                "p_diff_pre": [p_diff_pre],
    #                                "t_diff_post": [t_diff_post],
    #                                "p_diff_post": [p_diff_post]}
    #                               )
    diff_zpdes = ttest(mean_pre_zpdes, mean_post_zpdes, paired=True)
    diff_baseline = ttest(mean_pre_baseline, mean_post_baseline, paired=True)
    diff_pre = ttest(pd.to_numeric(mean_pre_zpdes, errors='coerce'),
                               pd.to_numeric(mean_pre_baseline, errors='coerce'), paired=False)
    diff_post = ttest(pd.to_numeric(mean_post_zpdes, errors='coerce'),
                                     pd.to_numeric(mean_post_baseline, errors='coerce'), paired=False)
    save = pd.DataFrame.from_dict({"t_diff_zpdes": [diff_zpdes["T"].values[0]],
                                   "p_diff_zpdes": [diff_zpdes["p-val"].values[0]],
                                   "d_diff_zpdes": [diff_zpdes["cohen-d"].values[0]],
                                   "BF_diff_zpdes": [diff_zpdes["BF10"].values[0]],
                                   "t_diff_baseline": [diff_baseline["T"].values[0]],
                                   "p_diff_baseline": [diff_baseline["p-val"].values[0]],
                                   "d_diff_baseline": [diff_baseline["cohen-d"].values[0]],
                                   "BF_diff_baseline": [diff_baseline["BF10"].values[0]],
                                   "t_diff_pre": [diff_pre["T"].values[0]],
                                   "p_diff_pre": [diff_pre["p-val"].values[0]],
                                   "d_diff_pre": [diff_pre["cohen-d"].values[0]],
                                   "BF_diff_pre": [diff_pre["BF10"].values[0]],
                                   "t_diff_post": [diff_post["T"].values[0]],
                                   "p_diff_post": [diff_post["p-val"].values[0]],
                                   "d_diff_post": [diff_post["cohen-d"].values[0]],
                                   "BF_diff_post": [diff_post["BF10"].values[0]],
                                   }
                                  )
    save.to_csv(f"{path_to_store}/info_aggregate-{normalization_type}-{study}.csv")


if __name__ == '__main__':
    # Open all needed dataset:
    all_studies = ["v3_prolific", "v3_utl", "v0_ubx", "v0_prolific", "v1_prolific", "v1_ubx", "v2_prolific"]
    # studies = ["v3_utl"]
    models = ["PCA", "ICA"]
    # expe_name = "all_studies_pre_test"
    # time_condition_to_keep = "PRE_TEST"
    global_expe_name = "probit-all-tasks-ordered"
    # pairs = [
    #          (["v3_prolific"], "PRE_TEST", f"v3_prolific_pre_test-{global_expe_name}"),
    #          (["v3_prolific"], "all", f"v3_prolific_all-{global_expe_name}"),
    #          (["v3_utl"], "PRE_TEST", f"v3_utl_pre_test-{global_expe_name}"),
    #          (["v3_utl"], "all", f"v3_utl_all-{global_expe_name}"),
    #          (["v0_ubx", "v0_prolific"], "PRE_TEST", f"v0_pre_test-{global_expe_name}"),
    #          (["v0_ubx", "v0_prolific"], "all", f"v0_all-{global_expe_name}"),
    #          (all_studies, "PRE_TEST", f"all_studies_pre_test-{global_expe_name}"),
    #          (all_studies, "all", f"all_studies_all-{global_expe_name}"),
    #          (all_studies, "PRE_TEST", f"all_minus_utl_studies_pre_test-{global_expe_name}")
    #          ]
    pairs = [
        # (["v3_prolific", "v3_utl"], "all", f"v3_studies_all-{global_expe_name}"),
        (["v3_prolific"], "PRE_TEST", f"v3_prolific_pre_test-{global_expe_name}"),
        (["v3_prolific"], "all", f"v3_prolific_all-{global_expe_name}"),
    ]
    # Kept columns:
    conditions_acc = {"moteval": ["moteval_1-speed", "moteval_4-speed", "moteval_8-speed"],
                      "enumeration": ['enumeration_9', 'enumeration_8', 'enumeration_6', 'enumeration_5',
                                      'enumeration_7'],
                      "loadblindness": ['loadblindness_far', 'loadblindness_near'],
                      "workingmemory": ['workingmemory_4', 'workingmemory_6', 'workingmemory_7', 'workingmemory_8',
                                        'workingmemory_5'],
                      "memorability": ['memorability_2-hit', 'memorability_3-hit', 'memorability_4-hit',

                                       'memorability_5-hit', 'memorability_100-hit']}
    conditions_rt = {
        "gonogo": ["gonogo_GO"],
        "taskswitch": ['taskswitch_parity-switching-cost', 'taskswitch_relative-switching-cost'],
        "ufov": ['ufov_final']
    }
    masks = {
        f"v3_studies_all-{global_expe_name}": {
            'PC1': ['workingmemory_4-accuracy', 'workingmemory_6-accuracy', 'workingmemory_7-accuracy',
                    'workingmemory_8-accuracy', 'workingmemory_5-accuracy', 'memorability_2-hit-accuracy',
                    'memorability_3-hit-accuracy', 'memorability_4-hit-accuracy', 'memorability_5-hit-accuracy'],
            'PC2': ['memorability_100-hit-accuracy'],
            'PC3': ['enumeration_9-accuracy', 'enumeration_8-accuracy', 'enumeration_6-accuracy',
                    'enumeration_5-accuracy',
                    'enumeration_7-accuracy'],
            'PC4': ['loadblindness_far-accuracy', 'loadblindness_near-accuracy', 'taskswitch_parity-switching-cost-rt',
                    'taskswitch_relative-switching-cost-rt', 'ufov_final-rt'],
            'PC5': ['loadblindness_far-accuracy', 'loadblindness_near-accuracy', 'taskswitch_parity-switching-cost-rt',
                    'taskswitch_relative-switching-cost-rt'],
            'PC6': ['moteval_1-speed-accuracy', 'moteval_4-speed-accuracy', 'moteval_8-speed-accuracy']
        }
    }
    masks = {}

    kept_columns = ["participant_id", "condition", "task_status", "study"]
    # Create all tasks (accuracy + RT type):
    tasks = [[f"{cdt}-accuracy" for cdt in task_condition] for task_condition in conditions_acc.values()]
    tasks += [[f"{cdt}-rt" for cdt in task_condition] for task_condition in conditions_rt.values()]
    # Flatten tasks into 1-d
    tasks = [item for sublist in tasks for item in sublist]
    # Create nb especially for accuracy metrics normalization:
    tasks_nb = [[f"{cdt}-nb" for cdt in task_condition] for task_condition in conditions_acc.values()]
    # tasks_nb += [[f"{cdt}-nb" for cdt in task_condition] for task_condition in conditions_rt.values()]
    # Flatten tasks into 1-d
    tasks_nb = [item for sublist in tasks_nb for item in sublist]
    # tasks = moteval + memorability + gonogo + loadblindness + taskswitch + workingmemory + enumeration
    # Loop over every interesting combinations:
    n_components = 24
    for (studies, time_condition_to_keep, expe_name) in pairs:
        df = get_df(studies)
        # With all data:
        df = df[kept_columns + tasks + tasks_nb]
        df.sort_values(by=['participant_id', 'task_status'], inplace=True)
        df_analysis = copy.deepcopy(df)
        # Filter time_conditions
        if time_condition_to_keep != "all":
            df = df[df['task_status'] == time_condition_to_keep]
        path_to_store = f"../outputs/DR_results/{expe_name}/"
        # For this df, we want to compute some aggregate metrics to understand the differences
        if time_condition_to_keep == "all":
            compute_broad_aggregate_difference(copy.deepcopy(df), path_to_store, tasks, studies)
            compute_broad_aggregate_difference(copy.deepcopy(df), path_to_store, tasks, studies,
                                               normalization_type="percentile_rank")
        for model_name in models:
            path = f"{path_to_store}/{model_name}/"
            Path(path).mkdir(parents=True, exist_ok=True)
            latent_df, average_loadings, main_model_loadings, main_variable_contributions, main_model, rmses, \
                correlations = run_dimensionality_reduction(df,
                                                            tasks,
                                                            tasks_nb,
                                                            n_components=n_components,
                                                            DR_type=model_name,
                                                            normalization_mode="probit")
            # Save results of the PCA:
            save_results(latent_df, n_components, model_name, tasks, main_variable_contributions, main_model, path,
                         average_loadings, main_model_loadings, rmses, correlations)
            # For now we are really interested in looking at enhancement for v3_prolific and v3_utl:
            v3_prolific = get_df(['v3_prolific'])
            get_diff_from_PCA(v3_prolific, main_model, tasks, tasks_nb, 'v3_prolific', path)
            v3_utl = get_df(['v3_utl'])
            get_diff_from_PCA(v3_utl, main_model, tasks, tasks_nb, 'v3_utl', path)
            # Do the same but with mask
            if expe_name in masks:
                v3_prolific = get_df(['v3_prolific'])
                masked_analysis(v3_prolific, main_model, tasks, tasks_nb, "v3_prolific", path)
                v3_utl = get_df(['v3_utl'])
                masked_analysis(v3_utl, main_model, tasks, tasks_nb, "v3_utl", path)
