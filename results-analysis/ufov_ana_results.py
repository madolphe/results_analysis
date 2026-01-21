from utils import retrieve_and_init_models, add_difference_pre_post, get_pymc_trace

from pathlib import Path
from utils import *

import pandas as pd
import numpy as np


def transform_to_list(row):
    # We divide by 99 to have max staircase = 1
    return 1 - np.mean(list(map(lambda x: int(x), row['difficulty_proposed'].split(',')))[-5:]) / 99


def get_threshold_last(row):
    frame_duration = 17.5
    return np.mean(list(map(lambda x: float(x)*frame_duration, row['measured_difficulties_frame_count'].split(',')))[-5:])


def get_final_step(row):
    return np.mean(list(map(lambda x: int(x), row['difficulty_proposed'].split(',')))[-5:])


def get_nb_trials(row):
    return len(row['difficulty_proposed'].split(','))


def format_data(path, save_lfa=False):
    # FIRST TREAT THE CSV AND PARSE IT TO DF
    task = "ufov"
    # csv_path = f"../outputs/{study}/results_{study}/{task}/{task}.csv"
    try:
        df = pd.read_csv(f"{path}/{task}.csv", sep=",")
    except FileNotFoundError:
        return pd.DataFrame({})
    base = ['participant_id', 'task_status', 'condition']
    df["final-accuracy_continuous"] = df.apply(lambda row: transform_to_list(row), axis=1)
    df["final_step"] = df.apply(lambda row: get_final_step(row), axis=1)
    df["nb_trials"] = df.apply(lambda row: get_nb_trials(row), axis=1)
    df["final-threshold"] = df.apply(lambda row: get_threshold_last(row), axis=1)
    df["final-rt"] = df.apply(lambda row: get_threshold_last(row), axis=1)
    condition_accuracy = ["final-accuracy_continuous", "final_step", "nb_trials", "final-threshold", "final-rt"]
    df = df[base + condition_accuracy]
    df = delete_uncomplete_participants(df)
    if save_lfa:
        df.to_csv(f"{path}/{task}_lfa.csv", index=False)
    return df


def fit_model(study, conditions_to_fit, model=None, model_type="pooled_model", save_lfa=False):
    task = "ufov"
    savedir = f"../outputs/{study}/results_{study}"
    path = f"{savedir}/{task}"
    Path(savedir).mkdir(parents=True, exist_ok=True)
    df = format_data(path, save_lfa=save_lfa)
    if model:
        get_pymc_trace(df, conditions_to_fit, task=task, model_object=model, model_type=model_type, study=study)
