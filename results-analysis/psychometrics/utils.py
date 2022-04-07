import os
import copy
import pandas as pd
import matplotlib.pyplot as plt


def create_dir(name):
    if not os.path.isdir(name):
        os.mkdir(name)


def filter_condition(df, condition):
    return df[df['condition'] == condition]


def format_questionnaire(df):
    participants_id = list(set(df['id']))
    participants_rows = []
    for participant_id in participants_id:
        participant_answers = df[df['id'] == participant_id]
        sessions_id = list(set(participant_answers['session_id']))
        for session_id in sessions_id:
            row_participant_session_id = {'id_participant': participant_id, 'session_id': session_id,
                                          'condition': participant_answers['condition'].iloc[0]}
            participant_answers_session = participant_answers[participant_answers['session_id'] == session_id]
            for id, ans in participant_answers_session.iterrows():
                # If component already in row just add the value
                if ans.component in row_participant_session_id:
                    row_participant_session_id[ans.component] = (row_participant_session_id[
                                                                     ans.component] + ans.value) / 2
                else:
                    row_participant_session_id[ans.component] = ans.value
            participants_rows.append(copy.deepcopy(row_participant_session_id))
    return pd.DataFrame(participants_rows)


def get_mean_std(df_baseline, df_zpdes):
    df_baseline_mean = df_baseline.groupby('session_id').mean()
    df_baseline_std = df_baseline.groupby('session_id').std()
    df_zpdes_mean = df_zpdes.groupby('session_id').mean()
    df_zpdes_std = df_zpdes.groupby('session_id').std()
    return df_baseline_mean, df_baseline_std, df_zpdes_mean, df_zpdes_std


def display_cols_value(df_baseline, df_zpdes, dir):
    df_baseline_mean, df_baseline_std, df_zpdes_mean, df_zpdes_std = get_mean_std(df_baseline, df_zpdes)
    for col in df_baseline_mean.columns:
        plot_scatter_serie(df_baseline_mean, df_baseline_std, df_zpdes_mean, df_zpdes_std, col, dir)


def plot_scatter_serie(df_baseline_mean, df_baseline_std, df_zpdes_mean, df_zpdes_std, col, dir):
    plt.errorbar(df_baseline_mean[col].index, df_baseline_mean[col], yerr=df_baseline_std[col], label='baseline',
                 color='blue')
    plt.scatter(df_baseline_mean[col].index, df_baseline_mean[col], color='blue')
    plt.scatter(df_zpdes_mean[col].index, df_zpdes_mean[col], color='red')
    plt.errorbar(df_zpdes_mean[col].index, df_zpdes_mean[col], yerr=df_zpdes_std[col], label='zpdes', color='red')
    plt.legend()
    plt.title(col)
    plt.savefig(f"{dir}/{col}.png")
    plt.close()