from data import get_data
from model import PooledModel
import pandas as pd
import numpy as np

tasks = get_data()


def create_csv_summary(rope=[-0.01, 0.01]):
    for group in ['zpdes', 'baseline']:
        csv = pd.DataFrame(
            columns=['task_name', 'condition', 'mean_pre', 'mean_post', 'mean_diff', 'pre_3_HDI', 'post_3_HDI',
                     'diff_3_HDI', 'sig_diff'])
        for task, (data, condition_list) in tasks.items():
            for condition in condition_list:
                summary_path = f"fixed_posterior/{task}/{task}_{group}_results/summary-{task}_{group}-{condition}.csv"
                try:
                    new_row = pd.read_csv(summary_path)
                except:
                    print("Path incorrect")
                pre_test = new_row[new_row['Unnamed: 0'] == "pre_test_posterior"]
                post_test = new_row[new_row['Unnamed: 0'] == "post_test_posterior"]
                diff = new_row[new_row['Unnamed: 0'] == "difference_of_means"]
                rope_in_diff_hdi = all(rope[1] <= diff['hdi_3%']) or all(rope[0] >= diff['hdi_97%'])
                tmp_row = {'task_name': task,
                           'condition': condition,
                           'mean_pre': pre_test['mean'].values[0],
                           'pre_3_HDI': pre_test['hdi_3%'].values[0],
                           'pre_97_HDI': pre_test['hdi_97%'].values[0],
                           'mean_post': post_test['mean'].values[0],
                           'post_3_HDI': post_test['hdi_3%'].values[0],
                           'post_97_HDI': post_test['hdi_97%'].values[0],
                           'mean_diff': diff['mean'].values[0],
                           'diff_3_HDI': diff['hdi_3%'].values[0],
                           'diff_97_HDI': diff['hdi_97%'].values[0],
                           'sig_diff': rope_in_diff_hdi
                           }
                csv = csv.append(tmp_row, ignore_index=True)
        csv.to_csv(f'summary_results_{group}.csv')


def add_rope_on_traces():
    for group in ['zpdes', 'baseline']:
        for task, (data, condition_list) in tasks.items():
            for condition in condition_list:
                path_traces = f"pooled_model/{task}/{task}_{group}_results/{task}_{group}-{condition}-trace"
                model = PooledModel(data, name=task, group=group, folder='pooled_model', stim_cond_list=condition_list,
                                    sample_size=500, traces_path=path_traces)
                model.condition = condition
                model.plot_estimated_posteriors()


def compute_BF_for_all_tasks():
    for group in ['zpdes', 'baseline']:
        csv = pd.DataFrame(columns=['task', 'condition', 'BF_mean'])
        for task, (data, condition_list) in tasks.items():
            for condition in condition_list:
                path_traces = f"pooled_model/{task}/{task}_{group}_results/{task}_{group}-{condition}-trace"
                model = PooledModel(data, name=task, group=group, folder='pooled_model', stim_cond_list=condition_list,
                                    sample_size=500, traces_path=path_traces)
                model.condition = condition
                BF_values = model.compute_effect_size(model.traces, 'difference_of_means', prior_odds=4.34)
                csv = csv.append({'task': task, 'condition': condition, 'BF_mean': np.mean(BF_values)}, ignore_index=True)
        csv.to_csv(f"BF_{group}_results.csv")


if __name__ == '__main__':
    # create_csv_summary()
    # add_rope_on_traces()
    compute_BF_for_all_tasks()
