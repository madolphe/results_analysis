import json
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import os

import enumeration_ana_results
import gonogo_ana_results
import taskswitch_ana_results
import moteval_ana_results
import memorability_ana_results
import loadblindness_ana_results
import workingmemory_ana_results

from pymc.utils import compute_diff_questionnaire_per_session, pairwise_comparison_questionnaire

from pymc.model import PooledModel, PooledModelRTSimulations, PooledModelRTCostSimulations, PowerAnalysis

study = 'v2_prolific'
# study = 'v1_prolific'
model_type = 'pooled_model'
with open('config/conditions.JSON', 'r') as f:
    all_conditions = json.load(f)


def run_all(study, accuracy_model=None, RT_model=None, accuracy_model_type=None, RT_model_type=None):
    if accuracy_model:
        enumeration_ana_results.fit_model(study=study, model=accuracy_model,
                                          conditions_to_fit=all_conditions["accuracy"]["enumeration"],
                                          model_type=accuracy_model_type)
        gonogo_ana_results.fit_model(study, model=accuracy_model,
                                     conditions_to_fit=all_conditions["accuracy"]["gonogo"],
                                     model_type=accuracy_model_type)
        taskswitch_ana_results.fit_model(study, model=accuracy_model,
                                         conditions_to_fit=all_conditions["accuracy"]["taskswitch"],
                                         model_type=accuracy_model_type)
        moteval_ana_results.fit_model(study, model=accuracy_model,
                                      conditions_to_fit=all_conditions["accuracy"]["moteval"],
                                      model_type=accuracy_model_type)
        memorability_ana_results.fit_model(study, conditions_to_fit=all_conditions["accuracy"]["memorability"],
                                           model=accuracy_model, model_type=accuracy_model_type)
        loadblindness_ana_results.fit_model(study, model=accuracy_model,
                                            conditions_to_fit=all_conditions["accuracy"]["loadblindness"],
                                            model_type=accuracy_model_type)
        workingmemory_ana_results.fit_model(study, model=accuracy_model,
                                            conditions_to_fit=all_conditions["accuracy"]["workingmemory"],
                                            model_type=accuracy_model_type)
    if RT_model:
        taskswitch_ana_results.fit_model(study, conditions_to_fit=all_conditions["RT"]["taskswitch"],
                                         model=PooledModelRTCostSimulations, model_type="pooled_model_RT")
        memorability_ana_results.fit_model(study, conditions_to_fit=all_conditions["RT"]["memorability"],
                                           model=PooledModelRTSimulations, model_type=RT_model_type)
        gonogo_ana_results.fit_model(study, conditions_to_fit=all_conditions["RT"]["gonogo"], model=RT_model,
                                     model_type=RT_model_type)


def init_from_traces_and_plot(study, accuracy_model=None, RT_model=None, accuracy_model_type=None, RT_model_type=None):
    if accuracy_model:
        gonogo_ana_results.run_visualisation(study, model=accuracy_model, model_type=accuracy_model_type,
                                             conditions_to_keep=all_conditions["accuracy"]["gonogo"])
        taskswitch_ana_results.run_visualisation(study, conditions_to_keep=all_conditions["accuracy"]["taskswitch"],
                                                 model=accuracy_model, model_type=accuracy_model_type)
        memorability_ana_results.run_visualisation(study, conditions_to_keep=all_conditions["accuracy"]["memorability"],
                                                   model=PooledModel, model_type=accuracy_model_type)
        enumeration_ana_results.run_visualisation(study=study, model=accuracy_model, model_type=accuracy_model_type,
                                                  conditions_to_keep=all_conditions["accuracy"]["enumeration"])
        moteval_ana_results.run_visualisation(study, model=accuracy_model, model_type=accuracy_model_type,
                                              conditions_to_keep=all_conditions["accuracy"]["moteval"])
        loadblindness_ana_results.run_visualisation(study, model=accuracy_model, model_type=accuracy_model_type,
                                                    conditions_to_keep=all_conditions["accuracy"]["loadblindness"])
        workingmemory_ana_results.run_visualisation(study, model=accuracy_model, model_type=accuracy_model_type,
                                                    conditions_to_keep=all_conditions["accuracy"]["workingmemory"])

    if RT_model:
        taskswitch_ana_results.run_visualisation(study, conditions_to_keep=all_conditions["RT"]["taskswitch"],
                                                 model=PooledModelRTCostSimulations, model_type=RT_model_type)
        memorability_ana_results.run_visualisation(study, conditions_to_keep=all_conditions["RT"]["memorability"],
                                                   model=RT_model, model_type=RT_model_type)
        gonogo_ana_results.run_visualisation(study, model=RT_model, model_type=RT_model_type,
                                             conditions_to_keep=all_conditions["RT"]["gonogo"])


def get_csv_for_all(study):
    enumeration_ana_results.fit_model(study=study, conditions_to_fit=all_conditions["accuracy"]["enumeration"],
                                      save_lfa=True)
    gonogo_ana_results.fit_model(study, conditions_to_fit=all_conditions["accuracy"]["gonogo"], save_lfa=True)
    taskswitch_ana_results.fit_model(study, conditions_to_fit=all_conditions["accuracy"]["taskswitch"], save_lfa=True)
    moteval_ana_results.fit_model(study, conditions_to_fit=all_conditions["accuracy"]["moteval"], save_lfa=True)
    # memorability_ana_results.fit_model(study, conditions_to_fit=all_conditions["accuracy"]["memorability"],
    #                                    save_lfa=True)
    loadblindness_ana_results.fit_model(study, conditions_to_fit=all_conditions["accuracy"]["loadblindness"],
                                        save_lfa=True)
    workingmemory_ana_results.fit_model(study, conditions_to_fit=all_conditions["accuracy"]["workingmemory"],
                                        save_lfa=True)
    # memorability_ana_results.fit_model(study, conditions_to_fit=all_conditions["RT"]["memorability"], save_lfa=True)


def run_questionnaires(study):
    dict_questionnaire = {
        'ues': (['engagement_score', 'AE', 'FA', 'PU', 'RW'], [0, 2, 4, 5, 7, 9]),
        'sims': (['Intrinsic_motivation', 'Identified_regulation', 'External_regulation', 'Amotivation'], [1, 4, 5, 8]),
        # 'sims': (['Amotivation'], [1, 4, 5, 8]),
        'tens': (['Competence', 'Autonomy'], [1, 4, 5, 8]),
        'nasa_tlx': (
            # ['Mental_demand', 'Physical_demand', 'Temporal_demand', 'Performance', 'Effort', 'Frustration',
            #  'load_index'],
            ['Mental_demand', 'Physical_demand', 'Temporal_demand', 'Performance', 'Effort', 'Frustration',
             'load_index'],
            [i for i in range(1, 8)])
    }
    for questionnaire, cdt_sessions in dict_questionnaire.items():
        pairwise_comparison_questionnaire(study, questionnaire)
        # compute_diff_questionnaire_per_session(questionnaire, cdt_sessions[0], cdt_sessions[1])


def get_raw_csv_for_study(study, root="../outputs"):
    # First get all csv
    enumeration_ana_results.format_data(f"{root}/{study}/results_{study}/enumeration", save_lfa=True)
    gonogo_ana_results.format_data(f"{root}/{study}/results_{study}/gonogo", save_lfa=True)
    taskswitch_ana_results.format_data(f"{root}/{study}/results_{study}/taskswitch", save_lfa=True)
    moteval_ana_results.format_data(f"{root}/{study}/results_{study}/moteval", save_lfa=True)
    memorability_ana_results.format_data(f"{root}/{study}/results_{study}/memorability", save_lfa=True)
    loadblindness_ana_results.format_data(f"{root}/{study}/results_{study}/loadblindness", save_lfa=True)
    workingmemory_ana_results.format_data(f"{root}/{study}/results_{study}/workingmemory", save_lfa=True)
    # Then merge everything into one df
    tasks = ['moteval', 'enumeration', 'gonogo', 'loadblindness', 'memorability', 'taskswitch', 'workingmemory']
    all_tasks = [pd.read_csv(f"{root}/{study}/results_{study}/{task}/{task}_lfa.csv") for task in tasks]
    common_parameters = ['participant_id', 'condition', 'task_status']
    for task_name, (ii, task_df) in zip(tasks, enumerate(all_tasks)):
        all_tasks[ii] = pd.DataFrame(
            {(f"{task_name}_{k}" if k not in common_parameters else k): v for k, v in task_df.items()})
    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on=common_parameters, how='outer'),
        all_tasks)
    kept_conditions = list(set([k for k in df_merged.keys() if ('accuracy' in k) or ('rt' in k)]))
    kept_conditions.remove("participant_id")  # participant_id contains 'rt' ^^
    df_merged = df_merged[common_parameters + kept_conditions]
    df_merged.to_csv(f"{root}/{study}/all_cognitive_results_{study}.csv")
    return df_merged


def get_frequentist_results(df, title, root):
    pre, post = df[df['task_status'] == 'PRE_TEST'], df[df['task_status'] == 'POST_TEST']
    participant_list = pd.unique(df['participant_id'])
    all_participants = {}
    for participant in participant_list:
        participant_dict = {}
        for condition in all_conditions['accuracy'].keys():
            for val in all_conditions['accuracy'][condition]:
                if f"{condition}_{val}-accuracy" in df.columns.values:
                    mean_diff = post[post['participant_id'] == participant][f"{condition}_{val}-accuracy"].iloc[0] - \
                                pre[pre['participant_id'] == participant][f"{condition}_{val}-accuracy"].iloc[0]
                    participant_dict[f"{condition}_{val}"] = mean_diff
        all_participants[participant] = participant_dict

    new_df = pd.DataFrame(all_participants)
    results = pd.DataFrame()
    results["p-val"] = new_df.apply(lambda row: stats.ttest_1samp(row.values, 0), axis=1)
    # This is useless, just to check that 1-sample (with mu_diff=0) was same as 2 samples ttests:
    ttests = {}
    for condition in all_conditions['accuracy'].keys():
        for val in all_conditions['accuracy'][condition]:
            if f"{condition}_{val}-accuracy" in df.columns.values:
                ttests[f"{condition}_{val}"] = stats.ttest_ind(post[f"{condition}_{val}-accuracy"],
                                                               pre[f"{condition}_{val}-accuracy"]).pvalue
    results['2samples-ind-pvalue'] = pd.Series(ttests)
    results["normality"] = new_df.apply(lambda row: stats.normaltest(row.values), axis=1)
    results["normality-sig"] = results.apply(lambda row: row["normality"].pvalue < 0.05, axis=1)
    results["sig"] = results.apply(lambda row: row["p-val"].pvalue < 0.05, axis=1)
    results['mean-diff'] = pd.DataFrame(all_participants).mean(axis=1)
    results['sd-diff'] = pd.DataFrame(all_participants).std(axis=1)
    results['effect-size'] = results['mean-diff'] / results['sd-diff']
    results.to_csv(f"{root}/{study}/{title}.csv")

    for index, participant in enumerate(new_df.columns):
        plt.scatter([i for i in range(len(new_df[participant].values))], new_df[participant].values, label=participant,
                    alpha=0.9, s=9)

    plt.errorbar(x=[i for i in range(len(results["mean-diff"].values))], y=results["mean-diff"].values,
                 yerr=results["sd-diff"].values, marker='*', color='black', fmt='o')
    plt.hlines(linestyle='--', xmin=0, xmax=len(new_df[participant].values), y=0, colors='black')
    x_ticks_label = []
    for index in new_df.index:
        if index in results[results['sig']].index.values:
            x_ticks_label.append(f"(*) {index}")
        else:
            x_ticks_label.append(index)
    plt.xticks([i for i in range(len(new_df[participant].values))], x_ticks_label, rotation=90)
    # plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    plt.tight_layout()
    plt.yticks(np.linspace(-1, 1, 21))
    plt.ylim(-1, 1)
    plt.rcParams["figure.figsize"] = (5, 7)
    plt.savefig(f"{root}/{study}/{title}.png")
    plt.close()


def frequentist_comparaison_of_2_samples(study1, study2, root):
    tasks = list(all_conditions['accuracy'].keys())
    if "comparison" not in os.listdir(f'{root}'):
        os.mkdir(f'{root}/comparison')
        for task in tasks:
            os.mkdir(f'{root}/comparison/{task}')
    study = pd.read_csv(f"{root}/{study1}/all_cognitive_results_{study1}.csv")
    study_ref = pd.read_csv(f"{root}/{study2}/all_cognitive_results_{study2}.csv")

    common_columns = list(set(study.columns).intersection(study_ref.columns))
    non_comparaison_cols = ['condition', 'task_status', 'participant_id', 'Unnamed: 0']
    results = {}
    for column in common_columns:
        if column not in non_comparaison_cols:
            results[column] = stats.ttest_ind(study[column].dropna(), study_ref[column].dropna())
    df = pd.DataFrame(results)
    for task in tasks:
        df[[col for col in df.columns if task in col]].to_csv(f"{root}/comparison/{task}/{study1}-vs-{study2}.csv")

    # Add conditions:
    if len(study['condition'].unique()) > 1 or len(study_ref['condition'].unique()) > 1:
        conditions_study = list(study['condition'].unique())
        conditions_study_ref = list(study_ref['condition'].unique())
        for condition in conditions_study:
            for condition_ref in conditions_study_ref:
                results = {}
                for column in common_columns:
                    if column not in non_comparaison_cols:
                        results[column] = stats.ttest_ind(study[study['condition'] == condition][column].dropna(),
                                                          study_ref[study_ref['condition'] == condition_ref][
                                                              column].dropna(), equal_var=False)
                df = pd.DataFrame(results)
                for task in tasks:
                    df[[col for col in df.columns if task in col]].to_csv(
                        f"{root}/comparison/{task}/uneq-var-{study1}_{condition}-vs-{study2}_{condition_ref}.csv")


if __name__ == '__main__':
    get_bayesian_models = False
    get_plots_from_trace = False
    get_questionnaires = False
    get_sample_size = False
    get_csv = False
    get_diff_results = True

    if get_bayesian_models:
        run_all(study, RT_model=PooledModelRTSimulations, RT_model_type="pooled_model_RT")
        run_all(study, accuracy_model=PooledModel, accuracy_model_type="pooled_model")

    if get_plots_from_trace:
        init_from_traces_and_plot(study, accuracy_model=PooledModel, accuracy_model_type="pooled_model")
        init_from_traces_and_plot(study, RT_model=PooledModelRTSimulations, RT_model_type="pooled_model_RT")

    if get_sample_size:
        # Sample size_estimation on MOT task:
        moteval_ana_results.fit_model(study, model=PowerAnalysis,
                                      conditions_to_fit=all_conditions["accuracy"]["moteval"],
                                      model_type="power_analysis")

    if get_questionnaires:
        run_questionnaires(study)

    # Get csv and get :
    studies = ["v2_prolific", "v1_prolific", "v1_ubx", "v0_ubx", "v0_prolific", "control_merged", "training_merged"]
    root_path = "../outputs/all_data_29_11_22"
    if get_csv:
        for study in studies:
            print(f'Study: {study}')
            df = get_raw_csv_for_study(study, root=root_path)
            get_frequentist_results(df, study, root=root_path)

    if get_diff_results:
        studies_to_compare = ["v2_prolific", "v1_prolific", "v1_ubx", "v0_ubx", "v0_prolific", "control_merged",
                              "training_merged"]
        for ii, study in enumerate(studies):
            if ii + 1 < len(studies):
                for study_to_compare in studies_to_compare[ii + 1:]:
                    frequentist_comparaison_of_2_samples(study, study_to_compare, root=root_path)