import pandas as pd
import copy
import matplotlib.pyplot as plt
from utils import *
from data import nasa, ues, sims, tens, lp
from statsmodels.graphics.factorplots import interaction_plot
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.patches as mpatches
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_breuschpagan
import os
import pymc as pm
import arviz as az
import pdb
import seaborn as sns

COLOR_ZPDES = "#114B5B"
COLOR_BASELINE = "#9CC19F"

COLOR_ZPDES_v3_prolific = "#114B5B"
COLOR_BASELINE_v3_prolific = "#9CC19F"

COLOR_ZPDES_v3_utl = "#114B5B"
COLOR_BASELINE_v3_utl = "#9CC19F"

COLOR_ZPDES_pilot = "#114B5B"
COLOR_BASELINE_pilot = "#9CC19F"


def clean_dataset(df):
    # Get the number of session, we take the median nb of sessions (avoid min and max if too much responses)
    nb_sessions = df.groupby('id_participant')['session_id'].count().median()
    correct_sessions_length = df.groupby('id_participant').filter(lambda x: len(x) == nb_sessions).iloc[0][
        "id_participant"]
    example_participant = df[df['id_participant'] == correct_sessions_length]['session_id']
    for p in df['id_participant'].unique():
        if len(df[df['id_participant'] == p]) > nb_sessions:
            print(f"Problem with particpant {p} - to many values")
        elif len(df[df['id_participant'] == p]) < nb_sessions:
            # Then retrieve which session was uncorrect
            participant_pb = df[df['id_participant'] == p]
            for id_ex in example_participant:
                if id_ex not in participant_pb['session_id'].values:
                    # Add row with the mean
                    row = pd.DataFrame([participant_pb.select_dtypes(include=['number']).drop(
                        columns=['id_participant', 'session_id']).mean()])
                    row['id_participant'] = p
                    row['session_id'] = id_ex
                    row['condition'] = participant_pb['condition'].iloc[0]
                    df = pd.concat([df, row], ignore_index=True)
    return df


def get_mean_per_week(df, threshlod_w0, nb_sessions):
    new_df = pd.DataFrame(columns=list(df.columns.values))
    for id_participant in set(df['id_participant']):
        tmp_part = df[df['id_participant'] == id_participant]
        if len(tmp_part) == nb_sessions:
            tmp_w0, tmp_w1 = tmp_part[tmp_part['session_id'] <= threshlod_w0], tmp_part[
                tmp_part['session_id'] > threshlod_w0]
            # tmp_w0['load_index'], tmp_w1['load_index'] = tmp_w0.loc[:, tmp_w0.columns != 'id_participant'].sum(
            #     axis=1), tmp_w1.loc[:, tmp_w0.columns != 'id_participant'].sum(axis=1)
            # w0, w1 = tmp_w0.mean(), tmp_w1.mean()
            w0, w1 = tmp_w0[tmp_w0.columns[tmp_w0.columns != "condition"]].mean(), tmp_w1[
                tmp_w1.columns[tmp_w1.columns != "condition"]].mean()
            w0['session_id'], w1['session_id'] = 0, 1
            w0['condition'], w1['condition'] = tmp_part['condition'].values[0], tmp_part['condition'].values[0]
            new_df = new_df._append(w0, ignore_index=True)
            new_df = new_df._append(w1, ignore_index=True)
    return new_df


def treat_nasa_tlx(df, study, plot=False):
    create_dir(f'nasa_tlx_{study}')
    questionnaire_name = f'nasa_tlx_{study}'
    conditions = ['Mental_demand', 'Physical_demand', 'Temporal_demand', 'Performance', 'Effort', 'Frustration',
                  'load_index']
    # Format questionnaire:
    df = format_questionnaire(df)
    # Drop of the 9th session
    df = df[df.loc[:, 'session_id'] != 9]
    df = df.groupby('id_participant').filter(lambda x: len(x) == 8)
    df = df.rename(columns={'Mental Demand': 'Mental_demand', 'Physical demand': 'Physical_demand',
                            'Temporal demand': 'Temporal_demand'})
    # df['load_index'] = df.loc[:, df.columns != 'id_participant'].sum(axis=1)
    df['load_index'] = df[conditions[:-1]].sum(axis=1)
    df = clean_dataset(df)
    df.to_csv(f'nasa_tlx_{study}/nasa_tlx_all.csv')

    if plot:
        # Plot boxplots for training questionnaires
        save_boxplots(df, conditions, questionnaire_name, additional_tag='all-', reverse=True)
        save_boxplots(df[df['condition'] == "zpdes"], conditions, questionnaire_name, additional_tag="zpdes-all-")
        save_boxplots(df[df['condition'] == "baseline"], conditions, questionnaire_name, additional_tag="baseline-all-")

    # Split and plot interractions
    df_tlx_baseline = filter_condition(df, 'baseline')
    df_tlx_zpdes = filter_condition(df, 'zpdes')
    # df_tlx_baseline['load_index'], df_tlx_zpdes['load_index'] = df_tlx_baseline.loc[:,
    #                                                             df_tlx_baseline.columns != 'id_participant'].sum(
    #     axis=1), df_tlx_zpdes.loc[:, df_tlx_zpdes.columns != 'id_participant'].sum(axis=1)
    # get 6 plots (one for each dims - 2 curves (ZPDES vs Baseline) through time
    if plot:
        display_cols_value(df_tlx_baseline, df_tlx_zpdes, 'nasa_tlx', study)
        possible_values = [i for i in range(0, 21)]
        display_reg_and_scatter(df, study, 'nasa_tlx', [c for c in conditions if c != 'load_index'],
                                xtickslabels=[f"s{i}" for i in range(1, 9)],
                                possible_values=possible_values, yticks=possible_values, ytickslabels=possible_values,
                                scatter=True)
        display_reg_and_scatter(df, study, 'nasa_tlx', ['load_index'], xtickslabels=[f"s{i}" for i in range(1, 9)],
                                possible_values=[i for i in range(0, 121, 10)], yticks=[i for i in range(0, 121, 10)],
                                ytickslabels=[i for i in range(0, 121, 10)],
                                scatter=True)
    new_df_baseline = get_mean_per_week(df_tlx_baseline, 4, 8)
    new_df_zpdes = get_mean_per_week(df_tlx_zpdes, 4, 8)
    new_df_zpdes.to_csv(f'nasa_tlx_{study}/nasa_tlx_zpdes_split.csv')
    new_df_baseline.to_csv(f'nasa_tlx_{study}/nasa_tlx_baseline_split.csv')
    diff_df_baseline = new_df_baseline.groupby('id_participant')[conditions].diff().dropna()
    diff_df_zpdes = new_df_zpdes.groupby('id_participant')[conditions].diff().dropna()
    diff_df_zpdes.to_csv(f'nasa_tlx_{study}/nasa_tlx_zpdes.csv')
    diff_df_baseline.to_csv(f'nasa_tlx_{study}/nasa_tlx_baseline.csv')
    # get 1 plot of the arithmetic mean
    # We want also to have format for JASP analysis:
    # For each participant, we want to have a row with all repeated measure i.e
    # participant_id, group, instrument_component, outcome_sess_0, ..., outcome_session_9
    new_df = get_JASP_format(df)
    new_df.to_csv(f'nasa_tlx_{study}/JASP_all.csv')


def treat_UES(df, study, plot=False):
    """
    Create exploratory graphs (boxplot and lines)
    Create different csv :
        - ues_all for every sessions
        - ues_group_split (x2) for only pre and post ues assessment
        - ues_diff_all_group (x2) for diff between week 0 and week 1 where w_0/1 == all sessions (including first one)
        - ues_group_diff_split (x2) for diff between week 0 and week 1 where w_0/1 == only first and final session
    """
    create_dir(f'ues_{study}')
    questionnaire_name = f'ues_{study}'
    conditions = ['FA-S.1', 'FA-S.2', 'FA-S.3', 'PU-S.1', 'PU-S.2', 'PU-S.3', 'AE-S.1', 'AE-S.2', 'AE-S.3', 'RW-S.1',
                  'RW-S.2', 'RW-S.3']
    conditions_to_keep = ['FA', 'PU', 'AE', 'RW', 'engagement_score']
    reverse_condition = ['PU-S.1', 'PU-S.2', 'PU-S.3']
    # Format questionnaire:
    df = format_questionnaire(df)
    df[reverse_condition] = 5 - df[reverse_condition]
    df['engagement_score'] = df[conditions].mean(axis=1)
    df['FA'] = df[['FA-S.1', 'FA-S.2', 'FA-S.3']].mean(axis=1)
    df['PU'] = df[['PU-S.1', 'PU-S.2', 'PU-S.3']].mean(axis=1)
    df['AE'] = df[['AE-S.1', 'AE-S.2', 'AE-S.3']].mean(axis=1)
    df['RW'] = df[['RW-S.1', 'RW-S.2', 'RW-S.3']].mean(axis=1)
    # Keep all sessions and groups:
    df = clean_dataset(df)
    all_df = df.drop(columns=conditions)
    all_df.to_csv(f'ues_{study}/ues_all.csv')
    pre_post_df = all_df.query('session_id == 0 | session_id ==9')
    if plot:
        # Plot boxplots for pre_post and for training questionnaires
        save_boxplots(all_df.query('session_id != 0 & session_id != 9'), conditions_to_keep, questionnaire_name,
                      additional_tag='training-', reverse=True)
        save_boxplots(all_df, conditions_to_keep, questionnaire_name, additional_tag="all-")
        save_boxplots(all_df[all_df['condition'] == "zpdes"], conditions_to_keep, questionnaire_name,
                      additional_tag="zpdes-all-")
        save_boxplots(all_df[all_df['condition'] == "baseline"], conditions_to_keep, questionnaire_name,
                      additional_tag="baseline-all-")
        save_boxplots(pre_post_df, conditions_to_keep, questionnaire_name, additional_tag="pre_post-")

    # Split df into zpdes / baseline
    all_df_baseline = filter_condition(all_df, 'baseline')
    all_df_zpdes = filter_condition(all_df, 'zpdes')
    pre_post_df_baseline = filter_condition(pre_post_df, 'baseline')
    pre_post_df_zpdes = filter_condition(pre_post_df, 'zpdes')
    pre_post_df_baseline.to_csv(f'ues_{study}/ues_baseline_split.csv')
    pre_post_df_zpdes.to_csv(f'ues_{study}/ues_zpdes_split.csv')

    # Display UES
    if plot:
        display_cols_value(all_df_baseline, all_df_zpdes, 'ues', study)
        df = all_df.query('session_id != 0 & session_id != 9')
        legend_labels = ["Strongly disagree", "Disagree", "Neither agree nor disagree", "Agree", "Strongly agree"]
        possible_values = [i for i in range(0, len(legend_labels))]
        display_reg_and_scatter(df, study, 'ues', conditions_to_keep, xtickslabels=["s2", "s4", "s5", "s7"],
                                possible_values=possible_values, yticks=possible_values, ytickslabels=possible_values,
                                scatter=True)
        # Values need to be round to use the stacked bar plot:
        df[conditions_to_keep] = df[conditions_to_keep].round().astype(int)
        colors_barplots = ['#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac']
        colors_barplots.reverse()
        display_reg_and_scatter(df, study, 'ues', conditions_to_keep, xtickslabels=["s1", "s4", "s5", "s8"],
                                possible_values=possible_values, yticks=[i / 10 for i in range(11)],
                                ytickslabels=[i * 10 for i in range(11)],
                                scatter=False, colors_barplots=colors_barplots, legend_labels=legend_labels)

    # Compute diff between weeks
    all_df_baseline = get_mean_per_week(all_df_baseline, 4, 6)
    all_df_zpdes = get_mean_per_week(all_df_zpdes, 4, 6)
    diff_df_baseline = all_df_baseline.groupby('id_participant')[conditions_to_keep].diff().dropna()
    diff_df_zpdes = all_df_zpdes.groupby('id_participant')[conditions_to_keep].diff().dropna()
    diff_df_zpdes.to_csv(f'ues_{study}/ues_diff_all_zpdes.csv')
    diff_df_baseline.to_csv(f'ues_{study}/ues_diff_all_baseline.csv')

    # Compute diff between pre_post:
    all_df_baseline = get_mean_per_week(pre_post_df_baseline, 4, 2)
    all_df_zpdes = get_mean_per_week(pre_post_df_zpdes, 4, 2)
    diff_df_baseline = all_df_baseline.groupby('id_participant')[conditions_to_keep].diff().dropna()
    diff_df_zpdes = all_df_zpdes.groupby('id_participant')[conditions_to_keep].diff().dropna()
    diff_df_zpdes.to_csv(f'ues_{study}/ues_diff_pre_post_zpdes.csv')
    diff_df_baseline.to_csv(f'ues_{study}/ues_diff_pre_post_baseline.csv')
    # For each participant, we want to have a row with all repeated measure i.e
    # participant_id, group, instrument_component, outcome_sess_0, ..., outcome_session_9
    new_df = get_JASP_format(df)
    new_df.to_csv(f'ues_{study}/JASP_all.csv')


def treat_SIMS(df, study, plot=False):
    create_dir(f'sims_{study}')
    questionnaire_name = f'sims_{study}'
    # Format questionnaire:
    df = format_questionnaire(df)
    df = df.rename(
        columns={'Intrinsic motivation': 'Intrinsic_motivation', 'Identified regulation': 'Identified_regulation',
                 'External regulation': 'External_regulation'})
    # Create SDI
    df['SDI'] = (2 * df['Intrinsic_motivation'] + df['Identified_regulation']) - (
                2 * df['Amotivation'] + df['External_regulation'])
    # keep all sessions and groups:
    df = df.groupby('id_participant').filter(lambda x: len(x) == 4)
    df = clean_dataset(df)
    df.to_csv(f'sims_{study}/sims_all.csv')

    # Split according to groups:
    df_sims_baseline = filter_condition(df, 'baseline')
    df_sims_zpdes = filter_condition(df, 'zpdes')
    conditions = ['Intrinsic_motivation', 'Identified_regulation', 'External_regulation', 'Amotivation', 'SDI']

    if plot:
        # Plot boxplots for training questionnaires
        save_boxplots(df, conditions, questionnaire_name, additional_tag='all-', reverse=True)
        save_boxplots(df[df['condition'] == "zpdes"], conditions, questionnaire_name, additional_tag="zpdes-all-")
        save_boxplots(df[df['condition'] == "baseline"], conditions, questionnaire_name, additional_tag="baseline-all-")
        # Plot evolution:
        display_cols_value(df_sims_baseline, df_sims_zpdes, 'sims', study)
        # We don't need the sdi for the stacked plots
        conditions = conditions[:-1]
        possible_values = [i for i in range(0, 7)]
        display_reg_and_scatter(df, study, 'sims', conditions, xtickslabels=["s1", "s4", "s5", "s8"],
                                possible_values=possible_values, yticks=possible_values, ytickslabels=possible_values,
                                scatter=True)
        possible_values_sdi = [i for i in range(-16, 17)]
        display_reg_and_scatter(df, study, 'sims', ["SDI"], xtickslabels=["s1", "s4", "s5", "s8"],
                                possible_values=possible_values, yticks=possible_values, ytickslabels=possible_values,
                                scatter=True)
        # Values need to be round to use the stacked bar plot:
        df[conditions] = df[conditions].round().astype(int)
        colors_barplots = ['#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac']
        colors_barplots.reverse()
        legend_labels = ["Not at all", "Very little", "A little", "Moderately", "Enough", "A lot", "Exactly"]
        display_reg_and_scatter(df, study, 'sims', conditions, xtickslabels=["s1", "s4", "s5", "s8"],
                                possible_values=possible_values, yticks=[i / 10 for i in range(11)],
                                ytickslabels=[i * 10 for i in range(11)],
                                scatter=False, colors_barplots=colors_barplots, legend_labels=legend_labels)
    new_df_baseline = get_mean_per_week(df_sims_baseline, 4, 4)
    new_df_zpdes = get_mean_per_week(df_sims_zpdes, 4, 4)
    new_df_zpdes.to_csv(f'sims_{study}/sims_zpdes_split.csv')
    new_df_baseline.to_csv(f'sims_{study}/sims_baseline_split.csv')
    diff_df_baseline = new_df_baseline.groupby('id_participant')[conditions].diff().dropna()
    diff_df_zpdes = new_df_zpdes.groupby('id_participant')[conditions].diff().dropna()
    diff_df_zpdes.to_csv(f'sims_{study}/sims_zpdes.csv')
    diff_df_baseline.to_csv(f'sims_{study}/sims_baseline.csv')
    # For each participant, we want to have a row with all repeated measure i.e
    # participant_id, group, instrument_component, outcome_sess_0, ..., outcome_session_9
    new_df = get_JASP_format(df)
    new_df.to_csv(f'sims_{study}/JASP_all.csv')


def treat_TENS(df, study, plot=False):
    create_dir(f'tens_{study}')
    questionnaire_name = f'tens_{study}'
    # Format questionnaire:
    df = format_questionnaire(df)
    conditions = ['Competence', 'Autonomy', 'Relatedness']
    # keep all sessions and groups:
    df = df.groupby('id_participant').filter(lambda x: len(x) == 4)
    df = clean_dataset(df)
    df.to_csv(f'tens_{study}/tens_all.csv')
    if plot:
        # Plot boxplots for training questionnaires
        save_boxplots(df, conditions, questionnaire_name, additional_tag='all-', reverse=True)
        save_boxplots(df[df['condition'] == "zpdes"], conditions, questionnaire_name, additional_tag="zpdes-all-")
        save_boxplots(df[df['condition'] == "baseline"], conditions, questionnaire_name, additional_tag="baseline-all-")

    # Split and plot: Plot evolution:
    df_tens_baseline = filter_condition(df, 'baseline')
    df_tens_zpdes = filter_condition(df, 'zpdes')
    if plot:
        display_cols_value(df_tens_baseline, df_tens_zpdes, 'tens', study)
        possible_values = [i for i in range(0, 6)]
        display_reg_and_scatter(df, study, 'tens', conditions, xtickslabels=["s1", "s4", "s5", "s8"],
                                possible_values=possible_values, yticks=possible_values, ytickslabels=possible_values,
                                scatter=True)
    new_df_baseline = get_mean_per_week(df_tens_baseline, 4, 4)
    new_df_zpdes = get_mean_per_week(df_tens_zpdes, 4, 4)

    # Store split results into csv
    new_df_zpdes.to_csv(f'tens_{study}/tens_zpdes_split.csv')
    new_df_baseline.to_csv(f'tens_{study}/tens_baseline_split.csv')
    diff_df_baseline = new_df_baseline.groupby('id_participant')[conditions].diff().dropna()
    diff_df_zpdes = new_df_zpdes.groupby('id_participant')[conditions].diff().dropna()
    diff_df_zpdes.to_csv(f'tens_{study}/tens_zpdes.csv')
    diff_df_baseline.to_csv(f'tens_{study}/tens_baseline.csv'),
    # For each participant, we want to have a row with all repeated measure i.e
    # participant_id, group, instrument_component, outcome_sess_0, ..., outcome_session_9
    new_df = get_JASP_format(df)
    new_df.to_csv(f'tens_{study}/JASP_all.csv')


def treat_LP(df, study, plot=True):
    create_dir(f'lp_{study}')
    questionnaire_name = f'lp_{study}'
    conditions = ['Difficulty_expectation', 'Easy_Feasible_Zone', 'Hard_Feasible_Zone']
    # 0: Lors de la prochaine activité d'entraînement, quel nombre de cibles souhaiteriez-vous avoir ?
    # 1: Quel est le nombre de cibles que vous pouvez suivre sans difficulté ?
    # 2: Quel est le nombre de cibles que vous pouvez suivre avec difficulté ?
    # Format questionnaire:
    # format_questionnaire() split according to component column
    # In LP questionnaire we need to split according to handle
    df['component'] = df['handle']
    df = format_questionnaire(df)
    df = clean_dataset(df)
    df = df.rename(columns={f'mot-LP-{i}': conditions[i] for i in range(3)})
    # Split and plot interractions
    df_baseline = filter_condition(df, 'baseline')
    df_zpdes = filter_condition(df, 'zpdes')
    if plot:
        # Plot boxplots for training questionnaires
        save_boxplots(df, conditions, questionnaire_name, additional_tag='all-', reverse=True)
        save_boxplots(df[df['condition'] == "zpdes"], conditions, questionnaire_name, additional_tag="zpdes-all-")
        save_boxplots(df[df['condition'] == "baseline"], conditions, questionnaire_name, additional_tag="baseline-all-")
        display_cols_value(df_baseline, df_zpdes, 'lp', study)
        possible_values = [i for i in range(2, 8)]
        display_reg_and_scatter(df, study, 'lp', conditions, xtickslabels=["s1", "s4", "s5", "s8"],
                                possible_values=possible_values, yticks=possible_values, ytickslabels=possible_values,
                                scatter=True)
        colors_barplots = ['#2166ac', '#67a9cf', '#d1e5f0', '#fddbc7', '#ef8a62', '#b2182b']
        display_reg_and_scatter(df, study, 'lp', conditions, xtickslabels=["s1", "s4", "s5", "s8"],
                                possible_values=possible_values, yticks=[i / 10 for i in range(11)],
                                ytickslabels=[i * 10 for i in range(11)],
                                scatter=False, colors_barplots=colors_barplots, legend_labels=possible_values)


def save_boxplots(df, conditions, instrument, additional_tag="", reverse=False):
    if not df.empty:
        for col in conditions:
            # Useless
            # fig = interaction_plot(x=df['session_id'].values, trace=df['condition'].values, response=df[col].values)
            # plt.savefig(f"{instrument}/{additional_tag}{col}_interraction.png")
            # plt.close()
            if not reverse:
                df.boxplot(column=col, by=['condition', 'session_id'])
            else:
                df.boxplot(column=col, by=['session_id', 'condition'])
            plt.xticks(rotation=45)
            plt.tight_layout()
            # plt.show()
            plt.savefig(f"{instrument}/{additional_tag}{col}_boxplot.png")
            plt.close()


def display_reg_and_scatter(df, study, dir, conditions, xtickslabels, possible_values, scatter=True, bpwidth=0.24,
                            yticks=[], ytickslabels=[], colors_barplots=[], legend_labels=[], plot_title=False,
                            grid_figure=(4, 2)):
    zpdes, baseline = filter_condition(df, 'zpdes'), filter_condition(df, 'baseline')
    nb_sessions = len(xtickslabels)
    # Plot the participant level:
    dict_groups = {"zpdes": zpdes, "baseline": baseline}
    used_labels = set()  # Keep track of labels that have already been added to the plot
    file_name = f"{dir}_{study}/stats_{conditions}.txt"
    file_name_bayesian = f"{dir}_{study}/stats_{conditions}_info_bayesian.txt"
    if scatter and os.path.isfile(file_name):
        os.remove(file_name)
    if scatter and os.path.isfile(file_name_bayesian):
        os.remove(file_name_bayesian)
    add_legend = True
    for condition in conditions:
        plt.figure(figsize=(8, 6), dpi=300)
        if scatter:
            zpdes_p, baseline_p, info_stats, info_bayesian = get_regression_line(
                df[['id_participant', 'session_id', 'condition', condition]],
                condition,
                range_max=np.max(possible_values), dir=dir, study=study)
            with open(file_name, 'a') as file:
                file.write(info_stats)
            with open(file_name_bayesian, 'a') as file:
                file.write(info_bayesian)
        for group_key, group_value in dict_groups.items():
            color = COLOR_ZPDES if group_key == "zpdes" else COLOR_BASELINE
            for p in group_value['id_participant'].unique():
                participant_values = group_value[group_value["id_participant"] == p][condition]
                base_shift = 0.15
                shift = base_shift if group_key == "zpdes" else -base_shift
                if scatter:
                    rd_deviation = np.random.random() / 10
                    plt.scatter([i + shift + rd_deviation for i in range(len(xtickslabels))], participant_values, s=1.5,
                                color=color,
                                alpha=1)
            if not scatter:
                # Get the proportion of each participant:
                proportion = group_value.groupby(['session_id', condition]).size().unstack(fill_value=0)
                proportion = proportion / proportion.sum(axis=1).max()
                # proportion = proportion.div(proportion.sum(axis=1), axis=0)
                # iterate over possiblities
                base_shift = 0.15
                shift = -base_shift if group_key == "zpdes" else base_shift
                bottom = np.zeros(nb_sessions)
                for i, val in enumerate(possible_values):
                    if val in proportion.columns:
                        label = legend_labels[i] if val not in used_labels else None  # Only label the first occurrence
                        plt.bar([i + shift for i in range(len(xtickslabels))], proportion[val], bottom=bottom,
                                label=label,
                                color=colors_barplots[i], width=bpwidth, alpha=0.8)
                        bottom += proportion[val]
                        used_labels.add(val)  # Mark this label as used
                # Juste for clarity add a border on the full bars:
                linestyle = '--' if group_key == "zpdes" else '-'
                plt.bar([i + shift for i in range(4)], [1 for i in range(len(xtickslabels))], color='none',
                        edgecolor=color,
                        width=bpwidth, linestyle=linestyle,
                        linewidth=1.5, label=group_key)
            if scatter:
                intercept, slope = zpdes_p if group_key == "zpdes" else baseline_p
                plt.plot([i for i in range(len(xtickslabels))],
                         [intercept + slope * i for i in range(len(xtickslabels))], c=color, label=group_key,
                         marker='s')
        if len(ytickslabels) > 0:
            plt.yticks(yticks, ytickslabels)
        plt.xticks([i for i in range(len(xtickslabels))], xtickslabels)
        if plot_title:
            plt.title(f"{condition}")
        if add_legend:
            legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            if not scatter:
                # Create a list of handles for the legend entries
                handles = [mpatches.Patch(facecolor='none', label='zpdes', edgecolor=COLOR_ZPDES, linestyle='--'),
                           mpatches.Patch(facecolor='none', label='baseline', edgecolor=COLOR_BASELINE)
                           ] + [mpatches.Patch(color=colors_barplots[i], label=legend_labels[i]) for i in
                                range(len(possible_values))]
                # Create the legend with the ordered handles and labels
                legend = plt.legend(loc='center left', handles=handles, bbox_to_anchor=(1, 0.5))
            add_legend = False
            plt.gca().add_artist(legend)
            plt.draw()
            legend.figure.savefig(f"{dir}_{study}/legend_scatter={scatter}.svg", dpi=300,
                                  bbox_inches=legend.get_window_extent().transformed(
                                      plt.gcf().dpi_scale_trans.inverted()))
            legend.remove()
        plt.legend('', frameon=False)
        plt.subplots_adjust(right=0.75)
        plt.savefig(f"{dir}_{study}/{condition}-scatter={scatter}.svg", bbox_inches='tight')
        plt.close()


def get_regression_line(df, condition, range_max, dir, study):
    '''Returns the regression parameters for each group + run an ANOVA groupxtime to assess significant diff'''
    zpdes, baseline = filter_condition(df, 'zpdes'), filter_condition(df, 'baseline')
    mdf_no_random_slopes_zpdes = smf.mixedlm(f"{condition} ~ session_id", zpdes, groups=zpdes["id_participant"]).fit()
    mdf_no_random_slopes_baseline = smf.mixedlm(f"{condition} ~ session_id", baseline,
                                                groups=baseline["id_participant"]).fit()
    zpdes_p = mdf_no_random_slopes_zpdes.params['Intercept'], mdf_no_random_slopes_zpdes.params['session_id']
    baseline_p = mdf_no_random_slopes_baseline.params['Intercept'], mdf_no_random_slopes_baseline.params['session_id']
    zpdes['group'] = 'zpdes'
    baseline['group'] = 'baseline'
    # Combine the dataframes
    combined_data = pd.concat([zpdes, baseline], ignore_index=True)
    # Define the mixed model with group, session_id, and their interaction
    model_formula = f"{condition} ~ group*session_id"
    mdf_combined = smf.mixedlm(model_formula, combined_data, groups=combined_data["id_participant"]).fit()
    shapiro_test = stats.shapiro(mdf_combined.resid)
    info = str(mdf_combined.summary()) + f"\n Shapiro test: {shapiro_test.statistic}, p={shapiro_test.pvalue} \n"
    info_bayesian = f"\n Bayesian analysis {condition}: \n" + get_bayesian_mixed_lm_model(combined_data, condition,
                                                                                          range_max, dir,
                                                                                          study)
    return zpdes_p, baseline_p, info, info_bayesian


def get_bayesian_mixed_lm_model(data, condition, range_max=20, dir='', study=''):
    # Converting categorical variables to codes
    data['group_code'] = data['group'].astype('category').cat.codes
    data['id_code'] = data['id_participant'].astype('category').cat.codes
    n_groups = len(data['id_participant'].unique())
    # n_sessions = len(data['session_id'].unique())
    with pm.Model() as model:
        # Priors
        intercept = pm.Normal('Intercept', mu=range_max, sigma=range_max / 10)
        group_slope = pm.Normal('group_slope', mu=0, sigma=1)
        session_id_slope = pm.Normal('session_id_slope', mu=0, sigma=1)
        group_session_interaction = pm.Normal('group_session_interaction', mu=0, sigma=1)
        sigma = pm.HalfNormal('sigma', sigma=10)
        participant_intercept = pm.Normal('participant_Intercept', mu=0, sigma=range_max / 10, shape=n_groups)
        # Expected value
        mu = (intercept +
              group_slope * data['group_code'] +
              session_id_slope * data['session_id'] +
              group_session_interaction * data['group_code'] * data['session_id'] +
              participant_intercept[data['id_code']])
        # Likelihood
        likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=data[condition])
        # Inference
        trace = pm.sample(10000, tune=1000, target_accept=0.95)
    generate_diagnostic_plots(model, trace, condition, dir, study)
    # Summary
    summary_df = az.summary(trace,
                            var_names=[var for var in trace.posterior.data_vars if var != 'participant_Intercept'])
    # Probability calculations
    rope = 0.001
    # A dictionary to hold the SDDR values
    sddr_values = {}
    # Define the parameters based on their priors in the model
    parameter_priors = {
        'group_slope': {'mu': 0, 'sigma': 1},
        'session_id_slope': {'mu': 0, 'sigma': 1},
        'group_session_interaction': {'mu': 0, 'sigma': 1},
    }
    summary_str = "Bayesian model summary:\n"
    for varname in ['Intercept', 'group_slope', 'session_id_slope', 'group_session_interaction']:
        flat_samples = trace.posterior[varname].values[:, 1000:].flatten()
        p_positive = (flat_samples > 0).mean()
        p_threshold = (abs(flat_samples) > rope).mean()
        summary_df.loc[varname, 'P(>0)'] = p_positive
        summary_df.loc[varname, f'P(||>{rope})'] = p_threshold
        direction = 'positive' if np.median(flat_samples) > 0 else 'negative'
        prob_in_direction = p_positive if direction == "positive" else 1 - p_positive
        summary_df.loc[varname, f'P(dir>{rope})'] = prob_in_direction
        # SDDR:
        if varname in parameter_priors:
            kde = stats.gaussian_kde(flat_samples)
            posterior_density_at_zero = kde(0)  # Evaluate the density at zero
            prior_density_at_zero = stats.norm.pdf(0, loc=parameter_priors[varname]['mu'],
                                                   scale=parameter_priors[varname]['sigma'])
            sddr = posterior_density_at_zero / prior_density_at_zero
            summary_df.loc[varname, 'SDDR'] = sddr
    summary_str += str(summary_df)
    return summary_str


def generate_diagnostic_plots(model, trace, title, dir, study, trace_plot=True, autocorr=False, rank=False):
    """
    Generate and save diagnostic plots for a Bayesian model trace.
    Parameters:
    - trace: MCMC trace from a PyMC model.
    - model_name: Name of the model, used for saving the figure.
    """
    if trace_plot:
        # Trace plots
        az.plot_trace(trace, var_names=['Intercept', 'group_slope', 'session_id_slope', 'group_session_interaction'])
        plt.tight_layout()
        plt.savefig(f"{dir}_{study}/{title}_trace.svg", format='svg', dpi=300)
        plt.close()
    if autocorr:
        # Autocorrelation plots
        az.plot_autocorr(trace, var_names=['Intercept', 'group_slope', 'session_id_slope', 'group_session_interaction'])
        plt.savefig(f"{dir}_{study}/{title}_autocorr.svg", format='svg', dpi=300)
        plt.close()
    if rank:
        # Rank plots
        az.plot_rank(trace)
        plt.tight_layout()
        plt.savefig(f"{dir}_{study}/{title}_rank.svg", format='svg', dpi=300)
        plt.close()


def plot_median_participant_baseline(trace):
    intercepts = trace.posterior['participant_Intercept']
    intercepts_df = intercepts.to_dataframe().reset_index()
    median_intercepts = intercepts_df.groupby('participant_Intercept_dim_0').median().reset_index()
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='participant_Intercept_dim_0', y='participant_Intercept', data=median_intercepts)
    plt.title('Distribution of Median Participant-Specific Intercepts')
    plt.xlabel('Participant ID')
    plt.ylabel('Median Intercept Value')
    plt.xticks(rotation=90)  # Rotate labels to avoid overlap
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    study = 'v3_prolific'
    treat_nasa_tlx(nasa, study=study, plot=True)
    treat_SIMS(sims, study=study, plot=True)
    treat_LP(lp, study=study, plot=True)
    treat_UES(ues, study=study, plot=True)
    treat_TENS(tens, study=study, plot=True)
