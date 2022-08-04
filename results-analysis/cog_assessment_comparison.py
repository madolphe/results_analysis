import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from typing import List


def get_data(study: str, study_ref: str, task_name: str, conditions=None):
    if not conditions:
        conditions = [f'{i}-accuracy' for i in range(5, 10)]
    df_prolific = pd.read_csv(f"../outputs/{study}/results_{study}/{task_name}/{task_name}_lfa.csv")
    df_ubx = pd.read_csv(f"../outputs/{study_ref}/results_{study_ref}/{task_name}/{task_name}_lfa.csv")
    results_prolific_zpdes = df_prolific[df_prolific['condition'] == 'zpdes'].groupby('participant_id')[
        conditions].diff().dropna()
    results_prolific_baseline = df_prolific[df_prolific['condition'] == 'baseline'].groupby('participant_id')[
        conditions].diff().dropna()
    results_ubx_zpdes = df_ubx[df_ubx['condition'] == 'zpdes'].groupby('participant_id')[
        conditions].diff().dropna()
    results_ubx_baseline = df_ubx[df_ubx['condition'] == 'baseline'].groupby('participant_id')[
        conditions].diff().dropna()
    return results_prolific_zpdes, results_prolific_baseline, results_ubx_zpdes, results_ubx_baseline


def plot_lines_ref_vs_new(study: str, task_name: str, all_conditions: List[str], results_new: pd.DataFrame,
                          results_ref: pd.DataFrame, title_group_condition: str, y_ticks_min=-0.5, y_ticks_max=0.8):
    plt.close()
    plt.axhline(y=0, color='r', linestyle='--')
    for row_ref in results_ref.iterrows():
        plt.plot([i for i in range(len(all_conditions))], row_ref[1].values, 'o-', c='grey', linewidth=0.5,
                 markersize=3)
    for row_new in results_new.iterrows():
        plt.plot([i for i in range(len(all_conditions))], row_new[1].values, 'o-', linewidth=1.75, markersize=10)
    if len(all_conditions) > 3:
        rotation = 20
        fontsize = 8
    else:
        rotation = 0
        fontsize = 10
    plt.xticks([i for i in range(len(all_conditions))], all_conditions, rotation=rotation, fontsize=fontsize)
    # plt.yticks(
    #     [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    plt.yticks(np.arange(y_ticks_min, y_ticks_max, 0.1))
    plt.margins()
    # [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.title(
        f"Difference between previous experiment \n and new participants in {title_group_condition} for task {task_name}")
    plt.savefig(f"outputs/{study}_{task_name}_{title_group_condition}.png")


if __name__ == '__main__':
    study_ref = 'v1_ubx'
    study = 'v1_prolific'
    task_name = 'moteval'
    with open('config/conditions_acc.JSON', 'r') as f:
        all_conditions = json.load(f)
    for task, conditions in all_conditions.items():
        results_prolific_zpdes, results_prolific_baseline, results_ubx_zpdes, \
        results_ubx_baseline = get_data(study, study_ref, task_name=task, conditions=conditions)
        plot_lines_ref_vs_new(study=study, task_name=task, all_conditions=conditions,
                              results_new=results_prolific_zpdes, results_ref=results_ubx_zpdes,
                              title_group_condition='ZPDES')
        plot_lines_ref_vs_new(study=study, task_name=task, all_conditions=conditions,
                              results_new=results_prolific_baseline, results_ref=results_ubx_baseline,
                              title_group_condition='BASELINE')
