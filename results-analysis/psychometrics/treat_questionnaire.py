import pandas as pd
import copy
import matplotlib.pyplot as plt
from utils import *
from data import nasa, ues, sims, tens


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


def treat_nasa_tlx(df):
    create_dir('nasa_tlx')
    conditions = ['Mental Demand', 'Physical demand', 'Temporal demand', 'Performance', 'Effort', 'Frustration',
                  'load_index']
    # Format questionnaire:
    df = format_questionnaire(df)
    # Drop of the 9th session
    df = df[df.loc[:, 'session_id'] != 9]
    df_tlx_baseline = filter_condition(df, 'baseline')
    df_tlx_zpdes = filter_condition(df, 'zpdes')
    df_tlx_baseline['load_index'], df_tlx_zpdes['load_index'] = df_tlx_baseline.loc[:,
                                                                df_tlx_baseline.columns != 'id_participant'].sum(
        axis=1), df_tlx_zpdes.loc[:, df_tlx_zpdes.columns != 'id_participant'].sum(axis=1)
    # get 6 plots (one for each dims - 2 curves (ZPDES vs Baseline) through time
    display_cols_value(df_tlx_baseline, df_tlx_zpdes, 'nasa_tlx')
    new_df_baseline = get_mean_per_week(df_tlx_baseline, 4, 8)
    new_df_zpdes = get_mean_per_week(df_tlx_zpdes, 4, 8)
    diff_df_baseline = new_df_baseline.groupby('id_participant')[conditions].diff().dropna()
    diff_df_zpdes = new_df_zpdes.groupby('id_participant')[conditions].diff().dropna()
    diff_df_zpdes.to_csv('nasa_tlx_zpdes.csv')
    diff_df_baseline.to_csv('nasa_tlx_baseline.csv')
    # get 1 plot of the arithmetic mean


def treat_UES(df):
    create_dir('ues')
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
    pre_post_df = df.query('session_id == 0 | session_id ==9')
    all_df_baseline = filter_condition(all_df, 'baseline')
    all_df_zpdes = filter_condition(all_df, 'zpdes')
    pre_post_df_baseline = filter_condition(pre_post_df, 'baseline')
    pre_post_df_zpdes = filter_condition(pre_post_df, 'zpdes')

    # Display UES
    display_cols_value(all_df_baseline, all_df_zpdes, 'ues')

    # Compute diff between weeks
    all_df_baseline = get_mean_per_week(all_df_baseline, 4, 6)
    all_df_zpdes = get_mean_per_week(all_df_zpdes, 4, 6)
    diff_df_baseline = all_df_baseline.groupby('id_participant')[conditions_to_keep].diff().dropna()
    diff_df_zpdes = all_df_zpdes.groupby('id_participant')[conditions_to_keep].diff().dropna()
    diff_df_zpdes.to_csv('ues/ues_diff_all_zpdes.csv')
    diff_df_baseline.to_csv('ues/ues_diff_all_baseline.csv')

    # Compute diff between pre_post:
    all_df_baseline = get_mean_per_week(pre_post_df_baseline, 4, 2)
    all_df_zpdes = get_mean_per_week(pre_post_df_zpdes, 4, 2)
    diff_df_baseline = all_df_baseline.groupby('id_participant')[conditions_to_keep].diff().dropna()
    diff_df_zpdes = all_df_zpdes.groupby('id_participant')[conditions_to_keep].diff().dropna()
    diff_df_zpdes.to_csv('ues/ues_diff_pre_post_zpdes.csv')
    diff_df_baseline.to_csv('ues/ues_diff_pre_post_baseline.csv')


def treat_SIMS(df):
    create_dir('sims')
    # Format questionnaire:
    df = format_questionnaire(df)
    df_sims_baseline = filter_condition(df, 'baseline')
    df_sims_zpdes = filter_condition(df, 'zpdes')
    conditions = ['Intrinsic motivation', 'Identified regulation', 'External regulation', 'Amotivation']
    # Plot evolution:
    display_cols_value(df_sims_baseline, df_sims_zpdes, 'sims')
    new_df_baseline = get_mean_per_week(df_sims_baseline, 4, 4)
    new_df_zpdes = get_mean_per_week(df_sims_zpdes, 4, 4)
    diff_df_baseline = new_df_baseline.groupby('id_participant')[conditions].diff().dropna()
    diff_df_zpdes = new_df_zpdes.groupby('id_participant')[conditions].diff().dropna()
    diff_df_zpdes.to_csv('sims/sims_zpdes.csv')
    diff_df_baseline.to_csv('sims/sims_baseline.csv')


def treat_TENS(df):
    create_dir('tens')
    # Format questionnaire:
    df = format_questionnaire(df)
    df_tens_baseline = filter_condition(df, 'baseline')
    df_tens_zpdes = filter_condition(df, 'zpdes')
    conditions = ['Competence', 'Autonomy']
    # Plot evolution:
    display_cols_value(df_tens_baseline, df_tens_zpdes, 'tens')
    new_df_baseline = get_mean_per_week(df_tens_baseline, 4, 4)
    new_df_zpdes = get_mean_per_week(df_tens_zpdes, 4, 4)
    diff_df_baseline = new_df_baseline.groupby('id_participant')[conditions].diff().dropna()
    diff_df_zpdes = new_df_zpdes.groupby('id_participant')[conditions].diff().dropna()
    diff_df_zpdes.to_csv('tens/tens_zpdes.csv')
    diff_df_baseline.to_csv('tens/tens_baseline.csv')


if __name__ == '__main__':
    # treat_nasa_tlx(nasa)
    # treat_UES(ues)
    # treat_SIMS(sims)
    treat_TENS(tens)
