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


def get_JASP_format(df):
    # Init an empty list; each elt will be a list with [participant_id, condition, sess_0, sess_i..., sess_n]
    # Get list of participants
    participants = list(df['id_participant'].unique())
    # Get list of subcomponents
    components = list(df.drop(['id_participant', 'session_id', 'condition'], axis=1).columns)
    # Get list of session_ids
    sessions_ids = list(df['session_id'].unique())
    all_participants = []
    # For participant in df, tmp_list = []
    for participant in participants:
        # For each component, tmp_list.append(participant_id, condition, sub_component)
        tmp_participant = df[df['id_participant'] == participant]
        for component in components:
            # For each session_id, tmp_list.append(df[participant_id, session_id, sub_component].value)
            tmp_list = [participant, tmp_participant['condition'].unique()[0], component]
            for sessions_id in sessions_ids:
                # When done; transform it into DF with columns names [participant_id, condition, sess_0, sess_i..., sess_n]
                val = tmp_participant.query(f"session_id=={sessions_id}")[component].values[0]
                tmp_list.append(val)
            all_participants.append(copy.deepcopy(tmp_list))
    new_df = pd.DataFrame(all_participants,
                          columns=['participant_id', 'group', 'component'] + [f"value_{sess_id}" for sess_id in
                                                                              sessions_ids])
    return new_df


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
