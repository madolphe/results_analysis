import pandas as pd
import copy
import matplotlib.pyplot as plt
from utils import *
from data import nasa, ues, sims, tens
from statsmodels.graphics.factorplots import interaction_plot


def get_mean_per_week(df, threshlod_w0, nb_sessions):
    new_df = pd.DataFrame(columns=list(df.columns.values))
    for id_participant in set(df['id_participant']):
        tmp_part = df[df['id_participant'] == id_participant]
        if len(tmp_part) == nb_sessions:
            tmp_w0, tmp_w1 = tmp_part[tmp_part['session_id'] <= threshlod_w0], tmp_part[
                tmp_part['session_id'] > threshlod_w0]
            # tmp_w0['load_index'], tmp_w1['load_index'] = tmp_w0.loc[:, tmp_w0.columns != 'id_participant'].sum(
            #     axis=1), tmp_w1.loc[:, tmp_w0.columns != 'id_participant'].sum(axis=1)
            w0, w1 = tmp_w0.mean(), tmp_w1.mean()
            w0['session_id'], w1['session_id'] = 0, 1
            w0['condition'], w1['condition'] = tmp_part['condition'].values[0], tmp_part['condition'].values[0]
            new_df = new_df.append(w0, ignore_index=True)
            new_df = new_df.append(w1, ignore_index=True)
    return new_df


def treat_nasa_tlx(df, study, plot=False):
    create_dir(f'nasa_tlx_{study}')
    questionnaire_name = 'nasa_tlx'
    conditions = ['Mental_demand', 'Physical_demand', 'Temporal_demand', 'Performance', 'Effort', 'Frustration',
                  'load_index']
    # Format questionnaire:
    df = format_questionnaire(df)
    # Drop of the 9th session
    df = df[df.loc[:, 'session_id'] != 9]
    df = df.groupby('id_participant').filter(lambda x: len(x) == 8)
    df = df.rename(columns={'Mental Demand': 'Mental_demand', 'Physical demand': 'Physical_demand',
                            'Temporal demand': 'Temporal_demand'})
    df['load_index'] = df.loc[:, df.columns != 'id_participant'].sum(axis=1)
    # Keep all sessions and groups:
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
        display_cols_value(df_tlx_baseline, df_tlx_zpdes, 'nasa_tlx')
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
    all_df = df.drop(columns=conditions)
    # Keep all sessions and groups:
    all_df.to_csv(f'ues_{study}/ues_all.csv')
    pre_post_df = all_df.query('session_id == 0 | session_id ==9')
    if plot:
        # Plot boxplots for pre_post and for training questionnaires
        save_boxplots(all_df.query('session_id != 0 & session_id != 9'), conditions_to_keep, 'ues',
                      additional_tag='training-', reverse=True)
        save_boxplots(all_df, conditions_to_keep, 'ues', additional_tag="all-")
        save_boxplots(all_df[all_df['condition'] == "zpdes"], conditions_to_keep, 'ues', additional_tag="zpdes-all-")
        save_boxplots(all_df[all_df['condition'] == "baseline"], conditions_to_keep, 'ues',
                      additional_tag="baseline-all-")
        save_boxplots(pre_post_df, conditions_to_keep, 'ues', additional_tag="pre_post-")

    # Split df into zpdes / baseline
    all_df_baseline = filter_condition(all_df, 'baseline')
    all_df_zpdes = filter_condition(all_df, 'zpdes')
    pre_post_df_baseline = filter_condition(pre_post_df, 'baseline')
    pre_post_df_zpdes = filter_condition(pre_post_df, 'zpdes')
    pre_post_df_baseline.to_csv(f'ues_{study}/ues_baseline_split.csv')
    pre_post_df_zpdes.to_csv(f'ues_{study}/ues_zpdes_split.csv')

    # Display UES
    if plot:
        display_cols_value(all_df_baseline, all_df_zpdes, 'ues')

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
    questionnaire_name = 'sims'
    # Format questionnaire:
    df = format_questionnaire(df)
    df = df.rename(
        columns={'Intrinsic motivation': 'Intrinsic_motivation', 'Identified regulation': 'Identified_regulation',
                 'External regulation': 'External_regulation'})

    # keep all sessions and groups:
    df = df.groupby('id_participant').filter(lambda x: len(x) == 4)
    df.to_csv(f'sims_{study}/sims_all.csv')

    # Split according to groups:
    df_sims_baseline = filter_condition(df, 'baseline')
    df_sims_zpdes = filter_condition(df, 'zpdes')
    conditions = ['Intrinsic_motivation', 'Identified_regulation', 'External_regulation', 'Amotivation']

    if plot:
        # Plot boxplots for training questionnaires
        save_boxplots(df, conditions, questionnaire_name, additional_tag='all-', reverse=True)
        save_boxplots(df[df['condition'] == "zpdes"], conditions, questionnaire_name, additional_tag="zpdes-all-")
        save_boxplots(df[df['condition'] == "baseline"], conditions, questionnaire_name, additional_tag="baseline-all-")
        # Plot evolution:
        display_cols_value(df_sims_baseline, df_sims_zpdes, 'sims')

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
    questionnaire_name = 'tens'
    # Format questionnaire:
    df = format_questionnaire(df)
    conditions = ['Competence', 'Autonomy']
    # keep all sessions and groups:
    df = df.groupby('id_participant').filter(lambda x: len(x) == 4)
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
        display_cols_value(df_tens_baseline, df_tens_zpdes, 'tens')
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


def save_boxplots(df, conditions, instrument, additional_tag="", reverse=False):
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


if __name__ == '__main__':
    study = 'v1_ubx'
    treat_nasa_tlx(nasa, study=study)
    treat_UES(ues, study=study)
    treat_SIMS(sims, study=study)
    treat_TENS(tens, study=study)
