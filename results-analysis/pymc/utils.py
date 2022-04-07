import copy

from data import get_data
from model import PooledModel, PooledModelRTSimulations, GLModel, PooledModelRTCostSimulations, \
    NormalNormalQuestionnaireModel
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg
import arviz as az
import pickle
import os

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


def get_RT_cost_posteriors():
    for task, (data, condition_list) in tasks.items():
        # For task switch - RT cost can be negative as it is cdt_switch - cdt_unswitch
        model_baseline = PooledModelRTCostSimulations(data=data, folder='RT_models_pooled', name=task, group='baseline',
                                                      stim_cond_list=condition_list, sample_size=8000)
        model_zpdes = PooledModelRTCostSimulations(data=data, folder='RT_models_pooled', name=task, group='zpdes',
                                                   stim_cond_list=condition_list, sample_size=8000)
        model_baseline.run(rope=(100, 100))
        model_zpdes.run(rope=(100, 100))
        model_zpdes.compare_traces(model_baseline.traces, 'difference_of_means')


def pairwise_comparison_questionnaire(questionnaire):
    data = pd.read_csv(f'../psychometrics/{questionnaire}/{questionnaire}_all.csv')
    # data = data.groupby('id_participant').filter(lambda x: len(x) == 6)
    baseline, zpdes = data.query('condition == "baseline"'), data.query('condition == "zpdes"')
    condition_list = list(zpdes.columns.drop(['Unnamed: 0', 'condition', 'session_id', 'id_participant']))
    session_id_list = zpdes['session_id'].unique().tolist()
    # session_id_list = [2, 4]
    for group, data in {'zpdes': zpdes, 'baseline': baseline}.items():
        model = NormalNormalQuestionnaireModel(data=data,
                                               folder='questionnaires_parwise', name=questionnaire,
                                               group=group,
                                               stim_cond_list=condition_list, sample_size=2000,
                                               session_id_list=session_id_list
                                               )
        model.run()


def compute_diff_questionnaire_per_session(questionnaire, conditions_list, sessions_id_list):
    if not os.path.isdir(f'questionnaires_parwise/{questionnaire}/group_diff_per_session'):
        os.mkdir(f'questionnaires_parwise/{questionnaire}/group_diff_per_session')
        os.mkdir(f'questionnaires_parwise/{questionnaire}/diff_btw_sessions')
        os.mkdir(f'questionnaires_parwise/{questionnaire}/group_diff_btw_sessions')
    common_path = f'questionnaires_parwise/{questionnaire}/'
    az_summary_metrics = ['condition', 'mean_mu', 'hdi_3%_mu', 'hdi_97%_mu', 'mean_sigm', 'hdi_3%_sigm', 'hdi_97%_sigm']
    df_btw_groups_per_sessions = pd.DataFrame(columns=['session_id'] + az_summary_metrics)
    df_btw_groups_btw_sessions = pd.DataFrame(columns=['session_id_0', 'session_id_1'] + az_summary_metrics)
    df_btw_sessions = pd.DataFrame(columns=['group', 'session_id_0', 'session_id_1'] + az_summary_metrics)
    for condition in conditions_list:
        for session_index, session_id in enumerate(sessions_id_list):
            baseline_0, zpdes_0 = get_traces_from_path_infos(common_path, condition, session_id)
            # Get the difference btw baseline & zpdes per session:
            diff_mu = baseline_0['traces'].posterior['mu'] - zpdes_0['traces'].posterior['mu']
            diff_sigm = baseline_0['traces'].posterior['sigma'] - zpdes_0['traces'].posterior['sigma']
            mu, sigm = az.summary(diff_mu), az.summary(diff_sigm)
            df_btw_groups_per_sessions = df_btw_groups_per_sessions.append({'condition': condition,
                                                                            'session_id': session_id,
                                                                            'mean_mu': mu['mean'].values[0],
                                                                            'hdi_3%_mu': mu['hdi_3%'].values[0],
                                                                            'hdi_97%_mu': mu['hdi_97%'].values[0],
                                                                            'mean_sigm': sigm['mean'].values[0],
                                                                            'hdi_3%_sigm': sigm['hdi_3%'].values[0],
                                                                            'hdi_97%_sigm': sigm['hdi_97%'].values[0]},
                                                                           ignore_index=True)
            # Get the difference betwen 2 sessions:
            if session_index + 1 < len(sessions_id_list):
                baseline_1, zpdes_1 = get_traces_from_path_infos(common_path, condition,
                                                                 sessions_id_list[session_index + 1])
                diff_within_groups_mu_baseline = baseline_1['traces'].posterior['mu'] - baseline_0['traces'].posterior[
                    'mu']
                diff_within_groups_sigm_baseline = baseline_1['traces'].posterior['sigma'] - \
                                                   baseline_0['traces'].posterior['sigma']
                diff_within_groups_mu_zpdes = zpdes_1['traces'].posterior['mu'] - zpdes_0['traces'].posterior[
                    'mu']
                diff_within_groups_sigm_zpdes = zpdes_1['traces'].posterior['sigma'] - zpdes_0['traces'].posterior[
                    'sigma']
                mu_baseline, sigm_baseline = az.summary(diff_within_groups_mu_baseline), az.summary(
                    diff_within_groups_sigm_baseline)
                mu_zpdes, sigm_zpdes = az.summary(diff_within_groups_mu_zpdes), az.summary(
                    diff_within_groups_sigm_zpdes)
                groups = ['baseline', 'zpdes']
                for ii, (group_mu, group_sig) in enumerate([(mu_baseline, sigm_baseline), (mu_zpdes, sigm_zpdes)]):
                    df_btw_sessions = df_btw_sessions.append({'group': groups[ii],
                                                              'condition': condition,
                                                              'session_id_0': session_id,
                                                              'session_id_1': sessions_id_list[session_index + 1],
                                                              'mean_mu': group_mu['mean'].values[0],
                                                              'hdi_3%_mu': group_mu['hdi_3%'].values[0],
                                                              'hdi_97%_mu': group_mu['hdi_97%'].values[0],
                                                              'mean_sigm': group_sig['mean'].values[0],
                                                              'hdi_3%_sigm': group_sig['hdi_3%'].values[0],
                                                              'hdi_97%_sigm': group_sig['hdi_97%'].values[0]},
                                                             ignore_index=True)
                diff_btw_grps_btw_sessions_mu = az.summary(diff_within_groups_mu_zpdes - diff_within_groups_mu_baseline)
                diff_btw_grps_btw_sessions_sigm = az.summary(
                    diff_within_groups_sigm_zpdes - diff_within_groups_sigm_baseline)
                df_btw_groups_btw_sessions = df_btw_groups_btw_sessions.append({
                    'condition': condition,
                    'session_id_0': session_id,
                    'session_id_1': sessions_id_list[session_index + 1],
                    'mean_mu': diff_btw_grps_btw_sessions_mu['mean'].values[0],
                    'hdi_3%_mu': diff_btw_grps_btw_sessions_mu['hdi_3%'].values[0],
                    'hdi_97%_mu': diff_btw_grps_btw_sessions_mu['hdi_97%'].values[0],
                    'mean_sigm': diff_btw_grps_btw_sessions_sigm['mean'].values[0],
                    'hdi_3%_sigm': diff_btw_grps_btw_sessions_sigm['hdi_3%'].values[0],
                    'hdi_97%_sigm': diff_btw_grps_btw_sessions_sigm['hdi_97%'].values[0]}, ignore_index=True
                )
    df_btw_groups_per_sessions.to_csv(common_path + f'group_diff_per_session/grp_diff_per_sess.csv')
    df_btw_sessions.to_csv(common_path + f'diff_btw_sessions/diff_btw_session.csv')
    df_btw_groups_btw_sessions.to_csv(common_path + f'group_diff_btw_sessions/grp_diff_btw_session.csv')


def get_traces_from_path_infos(common_path, condition, session_id):
    path_baseline = common_path + f'{questionnaire}_baseline_results/{questionnaire}_baseline-'
    path_zpdes = common_path + f'{questionnaire}_zpdes_results/{questionnaire}_zpdes-'
    tmp_path_baseline = path_baseline + f"{condition}-{session_id}-trace"
    tmp_path_zpdes = path_zpdes + f"{condition}-{session_id}-trace"
    # Load traces
    with open(tmp_path_baseline, 'rb') as buff:
        baseline = pickle.load(buff)
    with open(tmp_path_zpdes, 'rb') as buff:
        zpdes = pickle.load(buff)
    return baseline, zpdes


def treat_questionnaire():
    # questionnaires = ['ues', 'tens', 'sims']
    questionnaires = ['ues']
    # questionnaires = ['nasa_tlx']
    for questionnaire in questionnaires:
        print(questionnaire)
        baseline = pd.read_csv(f'questionnaires/{questionnaire}/{questionnaire}_baseline.csv')
        zpdes = pd.read_csv(f'questionnaires/{questionnaire}/{questionnaire}_zpdes.csv')
        baseline['condition'] = 'baseline'
        zpdes['condition'] = 'zpdes'
        df = pd.concat([baseline, zpdes])
        df = df.drop(columns=['Unnamed: 0'])
        condition_list = list(df.loc[:, df.columns != 'condition'].columns.values)
        # Anova results
        get_mixed_anova(questionnaire, ['engagement_score'])
        get_anova_results(questionnaire)
        # Bayesian models:
        model = GLModel(data=df, folder='questionnaires', name=questionnaire, group='all',
                        stim_cond_list=condition_list,
                        sample_size=8000)
        # model.run()
        print(f'DONE {questionnaire}')


def get_anova_results(questionnaire):
    return_csv = pd.DataFrame()
    baseline_split = pd.read_csv(f'questionnaires/{questionnaire}/{questionnaire}_baseline_split.csv')
    zpdes_split = pd.read_csv(f'questionnaires/{questionnaire}/{questionnaire}_zpdes_split.csv')
    baseline_split['condition'] = 'baseline'
    zpdes_split['condition'] = 'zpdes'
    df_split = pd.concat([baseline_split, zpdes_split])
    df_split = df_split.drop(columns=['Unnamed: 0'])
    condition_list = list(df_split.drop(columns=['session_id', 'condition', 'id_participant']).columns.values)
    for condition in condition_list:
        model = ols(f'{condition} ~ C(session_id) + C(condition) + C(session_id):C(condition)', data=df_split).fit()
        tmp_df = sm.stats.anova_lm(model, typ=2)
        tmp_df['subscale'] = condition
        tmp_df['questionnaire'] = questionnaire
        return_csv = return_csv.append(copy.deepcopy(tmp_df))
    return_csv.to_csv(f'questionnaires/{questionnaire}/anova_results.csv')


def get_mixed_anova(questionnaire, conditions):
    """
    Test functions for 2-way anova mixed
    """
    all_df = pd.read_csv(f'../psychometrics/{questionnaire}/{questionnaire}_all.csv')
    all_df = all_df.drop(columns=['Unnamed: 0', 'FA', 'PU', 'AE', 'RW']).query('session_id == 2 | session_id == 4')
    pg.mixed_anova(dv='engagement_score', between='condition', within='session_id', subject='id_participant',
                   data=all_df)


if __name__ == '__main__':
    # create_csv_summary()
    # add_rope_on_traces()
    # compute_BF_for_all_tasks()
    # get_mu_diff_group_csv()
    # get_RT_cost_posteriors()
    # treat_questionnaire()
    # pairwise_comparison_questionnaire()
    dict_questionnaire = {
        # 'ues': (['engagement_score', 'AE', 'FA', 'PU', 'RW'], [0, 2, 4, 5, 7, 9]),
        'sims': (['Intrinsic_motivation', 'Identified_regulation', 'External_regulation', 'Amotivation'], [1, 4, 5, 8]),
        # 'sims': (['Amotivation'], [1, 4, 5, 8]),
        'tens': (['Competence', 'Autonomy'], [1, 4, 5, 8]),
        'nasa_tlx': (
            # ['Mental_demand', 'Physical_demand', 'Temporal_demand', 'Performance', 'Effort', 'Frustration',
            #  'load_index'],
            ['Mental_demand', 'Physical_demand', 'Temporal_demand', 'Performance', 'Effort', 'Frustration', 'load_index'],
            [i for i in range(1, 8)])
    }
    for questionnaire, cdt_sessions in dict_questionnaire.items():
        # pairwise_comparison_questionnaire(questionnaire)
        compute_diff_questionnaire_per_session(questionnaire, cdt_sessions[0], cdt_sessions[1])
