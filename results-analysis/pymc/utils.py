from data import get_data
from model import PooledModel, PooledModelRTSimulations
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
    """ Method used once just to add rope on estimated posteriors (from older simulations)"""
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
                csv = csv.append({'task': task, 'condition': condition, 'BF_mean': np.mean(BF_values)},
                                 ignore_index=True)
        csv.to_csv(f"BF_{group}_results.csv")


def get_mu_diff_group_csv(rope=[-0.01, 0.01]):
    csv = pd.DataFrame(
        columns=['task_name', 'condition',
                 'pre_zpdes', 'post_zpdes', 'diff_zpdes',
                 'pre_baseline', 'post_baseline', 'diff_baseline',
                 'diff_pre_mean', 'diff_pre_HDI_3', 'diff_pre_HDI_97',
                 'diff_post_mean', 'diff_post_HDI_3', 'diff_post_HDI_97',
                 'diff_PG_mean', 'diff_PG_HDI_3', 'diff_PG_HDI_97', 'sig_diff'])
    for task, (data, condition_list) in tasks.items():
        for condition in condition_list:
            zpdes_model_path = f"pooled_model/{task}/{task}_zpdes_results/{task}_zpdes-{condition}-trace"
            baseline_model_path = f"pooled_model/{task}/{task}_baseline_results/{task}_baseline-{condition}-trace"
            model_zpdes = PooledModel(data, name=task, group='zpdes', folder='pooled_model',
                                      stim_cond_list=condition_list,
                                      sample_size=500, traces_path=zpdes_model_path)
            model_baseline = PooledModel(data, name=task, group='baseline', folder='pooled_model',
                                         stim_cond_list=condition_list,
                                         sample_size=500, traces_path=baseline_model_path)
            model_zpdes.condition, model_baseline.condition = condition, condition
            diff_pre = model_zpdes.compare_traces(model_baseline.traces, param_name='pre_test_theta')
            diff_post = model_zpdes.compare_traces(model_baseline.traces, param_name='post_test_theta')
            diff_PG = model_zpdes.compare_traces(model_baseline.traces, param_name='difference_of_means')
            zpdes_info = pd.read_csv(f"pooled_model/{task}/{task}_zpdes_results/{condition}-infos.csv").set_index(
                'Unnamed: 0')
            baseline_info = pd.read_csv(f"pooled_model/{task}/{task}_baseline_results/{condition}-infos.csv").set_index(
                'Unnamed: 0')
            rope_in_diff_hdi = all(rope[1] <= diff_PG['hdi_3%']) or all(rope[0] >= diff_PG['hdi_97%'])
            tmp_row = {'task_name': task,
                       'condition': condition,
                       'pre_zpdes': zpdes_info.loc['pre_test_theta', 'mean'],
                       'post_zpdes': zpdes_info.loc['post_test_theta', 'mean'],
                       'diff_zpdes': zpdes_info.loc['difference_of_means', 'mean'],
                       'pre_baseline': baseline_info.loc['pre_test_theta', 'mean'],
                       'post_baseline': baseline_info.loc['post_test_theta', 'mean'],
                       'diff_baseline': baseline_info.loc['difference_of_means', 'mean'],
                       'diff_pre_mean': diff_pre['mean'].values[0],
                       'diff_pre_HDI_3': diff_pre['hdi_3%'].values[0],
                       'diff_pre_HDI_97': diff_pre['hdi_97%'].values[0],
                       'diff_post_mean': diff_post['mean'].values[0],
                       'diff_post_HDI_3': diff_post['hdi_3%'].values[0],
                       'diff_post_HDI_97': diff_post['hdi_97%'].values[0],
                       'diff_PG_mean': diff_PG['mean'].values[0],
                       'diff_PG_HDI_3': diff_PG['hdi_3%'].values[0],
                       'diff_PG_HDI_97': diff_PG['hdi_97%'].values[0],
                       'sig_diff': rope_in_diff_hdi
                       }
            csv = csv.append(tmp_row, ignore_index=True)
    csv.to_csv(f'summary_results_diff.csv')


def get_RT_posteriors():
    for task, (data, condition_list) in tasks.items():
        model_baseline = PooledModelRTSimulations(data=data, folder='RT_models_pooled', name=task, group='baseline',
                                                  stim_cond_list=condition_list, sample_size=8000)
        model_zpdes = PooledModelRTSimulations(data=data, folder='RT_models_pooled', name=task, group='zpdes',
                                               stim_cond_list=condition_list, sample_size=8000)
        model_baseline.run()
        model_zpdes.run()
        model_zpdes.compare_traces(model_baseline.traces, 'difference_of_means')


if __name__ == '__main__':
    # create_csv_summary()
    # add_rope_on_traces()
    # compute_BF_for_all_tasks()
    # get_mu_diff_group_csv()
    get_RT_posteriors()
