import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import numpy as np


def get_data(study: str, study_ref: str, questionnaire_name: str, sessions_ids: List[str]):
    df_new = pd.read_csv(f"{study}/{questionnaire_name}/{questionnaire_name}_all.csv")
    df_ref = pd.read_csv(f"{study_ref}/{questionnaire_name}/{questionnaire_name}_all.csv")
    df_new = df_new[df_new.groupby('id_participant')['id_participant'].transform('size') == len(sessions_ids)]
    df_ref = df_ref[df_ref.groupby('id_participant')['id_participant'].transform('size') == len(sessions_ids)]
    return df_new, df_ref


def plot_ref_vs_new_per_component(df_ref: pd.DataFrame, df_new: pd.DataFrame, component_name: str,
                                  questionnaire_name: str, group: str, session_ids: List[str]):
    plt.figure()
    plt.tight_layout()
    # plt.axhline(y=0, color='r', linestyle='--')
    for participant in df_ref['id_participant'].unique():
        plt.plot([i for i in range(len(session_ids))],
                 df_ref.query(f'id_participant == {participant}')[component_name].values, 'o-', c='grey', linewidth=0.5,
                 markersize=3)
    for participant in df_new['id_participant'].unique():
        plt.plot([i for i in range(len(session_ids))],
                 df_new.query(f'id_participant == {participant}')[component_name].values, 'o-', linewidth=1.75,
                 markersize=10)
    fontsize = 8
    plt.xticks([i for i in range(len(session_ids))], session_ids, fontsize=fontsize)
    plt.yticks(np.arange(0, 8, 1))
    plt.title(
        f"Difference between previous experiment \n and new participants in {group} for questionnaire {questionnaire_name}\n and component {component_name}",
        fontsize=8)
    plt.savefig(f"outputs/{study}_{questionnaire_name}_{component_name}_{group}.png")
    plt.close()


if __name__ == '__main__':
    study_ref = 'v1_ubx'
    study = 'v1_prolific'
    # questionnaire_name = 'nasa_tlx'
    # questionaires = ['ues', 'nasa_tlx', 'sims', 'tens']
    # scales: ues=5, nasa=20, sims=7, tens=5
    questionaires = ['sims']
    with open('questionnaires.JSON', 'r') as f:
        questionnaires_details = json.load(f)
    # for questionnaire_name in questionnaires_details.keys():
    for questionnaire_name in questionaires:
        df_new, df_ref = get_data(study=study, study_ref=study_ref, questionnaire_name=questionnaire_name,
                                  sessions_ids=questionnaires_details[questionnaire_name]['session_ids'])
        df_ref_zpdes, df_ref_baseline = df_ref[df_ref["condition"] == 'zpdes'], df_ref[
            df_ref["condition"] == 'baseline']
        df_new_zpdes, df_new_baseline = df_new[df_new["condition"] == 'zpdes'], df_new[
            df_new["condition"] == 'baseline']
        for compononent in questionnaires_details[questionnaire_name]['components']:
            session_ids = questionnaires_details[questionnaire_name]['session_ids']
            plot_ref_vs_new_per_component(df_ref=df_ref_zpdes, df_new=df_new_zpdes, component_name=compononent,
                                          questionnaire_name=questionnaire_name, group='ZPDES', session_ids=session_ids)
            plot_ref_vs_new_per_component(df_ref=df_ref_baseline, df_new=df_new_baseline, component_name=compononent,
                                          questionnaire_name=questionnaire_name, group='BASELINE',
                                          session_ids=session_ids)
