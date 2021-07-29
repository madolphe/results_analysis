from scipy.stats import ttest_rel, shapiro, levene
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns
import numpy as np

dataframe = pd.read_csv("../outputs/moteval/moteval_treat.csv")

# We have to take all conditions and do every tests
# Correct the p-value with Bonferroni
# https://egap.org/resource/10-things-to-know-about-multiple-comparisons/

# First decision treat the mean over the 3 conditions for now:
# Q: Should we use compare PRE & POST for each condition ?
dataframe['mean_accuracy'] = dataframe[['1-accuracy', '4-accuracy', '8-accuracy']].mean(axis=1)
dataframe['mean_RT'] = dataframe[['1-RT', '4-RT', '8-RT']].mean(axis=1)
dataframe = dataframe[['task_status', 'mean_accuracy', 'mean_RT', 'participant_id']]


def plot_boxplot(dataframe, save_name="../outputs/qq_histo", save=True, show=True):
    fig, axs = plt.subplots(1, len(dataframe.drop(columns=['task_status', 'participant_id']).columns))
    overall_mean = dataframe.groupby(["participant_id"]).mean()
    overall_mean['task_status'] = 'GLOBAL'
    overall_mean['participant_id'] = overall_mean.index
    dataframe = dataframe.append(overall_mean)
    # For each outcome:
    for idx, col in enumerate(dataframe.drop(columns=['task_status', 'participant_id']).columns):
        sns.boxplot(x='task_status', y=col, data=dataframe[['task_status', col]], ax=axs[idx])
        sns.swarmplot(x='task_status', y=col, data=dataframe[['task_status', col]], color='black', ax=axs[idx])
    fig.tight_layout()
    fig.show()
    return


def plot_histo_and_qq(dataframe, save_name="../outputs/qq_histo", save=True, show=True):
    """
        Plot two lines : Pre and Post tests
        On each line: Histogram for each condition + QQ plots for each condition (i.e nb of condition x 2)
    """
    pretest = dataframe[dataframe['task_status'] == "PRE_TEST"]
    posttest = dataframe[dataframe['task_status'] == "POST_TEST"]
    fig, axs = plt.subplots(2, len(dataframe.drop(columns=['task_status', 'participant_id']).columns) * 2)
    for row_idx, data in enumerate([pretest, posttest]):
        # First plot histograms for each condition:
        idx_col = 0
        for col_name in data.columns.drop(['task_status', 'participant_id']):
            axs[row_idx, idx_col].hist(data[col_name])
            qqplot(data[col_name], line='s', ax=axs[row_idx, idx_col + 1])
            axs[row_idx, idx_col + 1].set_xlabel('')
            axs[row_idx, idx_col + 1].set_ylabel('')
            idx_col += 2
    fig.tight_layout()
    if show:
        fig.show()
    if save:
        fig.savefig(save_name)


plot_histo_and_qq(dataframe, save_name="../outputs/moteval_normality")
plot_boxplot(dataframe, save_name="../outputs/moteval_distrib")


# We should look at mean - variances - quartiles [boxplot] + histogram + QQPlotv+ normality-tests
def test_normality():
    pretest = dataframe[dataframe['task_status'] == "PRE_TEST"]
    posttest = dataframe[dataframe['task_status'] == "POST_TEST"]
    print("Shapiro normality test; H0=\"Normaly distributed samples\"")
    for col in pretest.columns:
        if 'accuracy' in col or 'RT' in col:
            # For each task day (pre&post) plot histo (title=normality-test) + QQplots (plot 2 cols)
            for idx, task_order in enumerate([pretest, posttest]):
                shapiro_outputs = shapiro(task_order[col])
                print(f"Session {idx} normality test= {shapiro_outputs}")
                if shapiro_outputs[1] < 0.05:
                    print("Normality hypothesis can be rejected")
                # plt.hist(task_order[col])
                # plt.show()
            print("\n")
    print("Be careful, small sample size => "
          "Normality test might lack power to detect the deviation of the variable from normality")
    print("\n")


# If we assume that normality is OK

# We should look at variances, are they equals ?
def variance_test(dataframe):
    pretest = dataframe[dataframe['task_status'] == "PRE_TEST"]
    posttest = dataframe[dataframe['task_status'] == "POST_TEST"]
    print("Levene tests; H0=\"sig1 = sig2 \"")
    for col in pretest.columns:
        if 'accuracy' in col or 'RT' in col:
            stats = levene(pretest[col].values, posttest[col].values)
            print(f'std: pre-test={np.std(pretest[col])}, post-test={np.std(posttest[col])}')
            if stats[1] < 0.05:
                print("Reject null hypothesis, difference between variance is statistically significant.")
            else:
                print("Can't reject null hypothesis.")
            print(f"{col} : {stats} \n ")


variance_test(dataframe)

print('--------------')


# Then we can ask ourselves: are the means equals?
def t_test(dataframe):
    pretest = dataframe[dataframe['task_status'] == "PRE_TEST"]
    posttest = dataframe[dataframe['task_status'] == "POST_TEST"]
    print("Paired t-tests; H0=\"mean_pre_test = mean_post_test \"")
    for col in pretest.columns:
        if 'accuracy' in col or 'RT' in col:
            print(f'means: pre-test={np.mean(pretest[col])}, post-test={np.mean(posttest[col])}')
            stats = ttest_rel(pretest[col].values, posttest[col].values, nan_policy="omit", alternative="two-sided")
            if stats[1] < 0.05:
                print("Reject null hypothesis, means have high chance to be equals.")
            else:
                print("Can't reject null hypothesis.")
            print(f"{col} : {stats} \n ")


t_test(dataframe)

# If it is not the case, then, what's the effect size? How to quantify it ?
# Compute Cohen's D for the paired samples
# Compute R2
