import os
import copy
import pandas as pd
import matplotlib.pyplot as plt

COLOR_ZPDES = "#114B5B"
COLOR_BASELINE = "#9CC19F"

COLOR_ZPDES_v3_prolific = "#114B5B"
COLOR_BASELINE_v3_prolific = "#9CC19F"

COLOR_ZPDES_v3_utl = "#114B5B"
COLOR_BASELINE_v3_utl = "#9CC19F"

COLOR_ZPDES_pilot = "#114B5B"
COLOR_BASELINE_pilot = "#9CC19F"


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
        if len(tmp_participant['session_id']) == len(sessions_ids):
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
    df_baseline_mean = df_baseline[df_baseline.columns[df_baseline.columns != 'condition']].groupby('session_id').mean()
    df_baseline_std = df_baseline[df_baseline.columns[df_baseline.columns != 'condition']].groupby('session_id').std()
    df_zpdes_mean = df_zpdes[df_zpdes.columns[df_zpdes.columns != 'condition']].groupby('session_id').mean()
    df_zpdes_std = df_zpdes[df_zpdes.columns[df_zpdes.columns != 'condition']].groupby('session_id').std()
    return df_baseline_mean, df_baseline_std, df_zpdes_mean, df_zpdes_std


def display_cols_value(df_baseline, df_zpdes, dir, study):
    for col in df_baseline.columns:
        if col != 'session_id' and col != 'condition' and col != 'id_participant':
            plot_scatter_serie(df_baseline, df_zpdes, col, dir, study)


def plot_scatter_serie(df_baseline, df_zpdes, col, dir, study):
    # Plot
    df_baseline_mean, df_baseline_std, df_zpdes_mean, df_zpdes_std = get_mean_std(df_baseline, df_zpdes)
    # Plot the participant level:
    tmp_df_dict = {"baseline": df_baseline, "zpdes": df_zpdes}
    for group, df in tmp_df_dict.items():
        for p in df['id_participant'].unique():
            participant_values = df[df["id_participant"] == p][col]
            color = COLOR_ZPDES if group == "zpdes" else COLOR_BASELINE
            base_shift = 0.2
            shift = base_shift if group == "zpdes" else -base_shift
            plt.scatter(df_baseline_mean[col].index.values + shift, participant_values, s=1.5, color=color, alpha=1)
    # Plot the group level statistics:
    shift = 0
    # plt.errorbar(df_baseline_mean[col].index.values - shift, df_baseline_mean[col], yerr=df_baseline_std[col], label='Baseline',
    #              color=COLOR_BASELINE, elinewidth=3, linewidth=3, alpha=0.9)
    plt.plot(df_baseline_mean[col].index.values - shift, df_baseline_mean[col], color=COLOR_BASELINE, alpha=1,
             marker="s", markersize=4, label='Baseline')
    # plt.errorbar(df_zpdes_mean[col].index.values + shift, df_zpdes_mean[col], yerr=df_zpdes_std[col], label='Zpdes',
    #              color=COLOR_ZPDES, elinewidth=3, linewidth=3, alpha=0.9)
    plt.plot(df_zpdes_mean[col].index.values + shift, df_zpdes_mean[col], color=COLOR_ZPDES, alpha=1, marker="s",
             markersize=4, label='ZPDES')
    # plt.legend(loc="center left", bbox_to_anchor=(1.3, 0.5))
    plt.title(col)
    plt.savefig(f"{dir}_{study}/{col}.png", bbox_inches='tight')
    plt.close()
