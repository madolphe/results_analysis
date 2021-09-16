import pandas as pd
from sklearn.decomposition import PCA

tasks_names = ['gonogo', 'moteval', 'taskswitch', 'loadblindness', 'memorability', 'enumeration', 'workingmemory']
import os

print(os.getcwd())
df_dict = {}
for task_name in tasks_names:
    df_dict[task_name] = pd.read_csv(f"../../outputs/{task_name}/{task_name}_lfa.csv")

# Specific adjustements:
# gonogo, moteval, loadblindness have already the lowest nb of columns

# Task switch:
# Drop nb trials and nb of correct in task switch:
for col in df_dict['taskswitch'].columns:
    if col != 'participant_id':
        if 'rt' not in col and 'accuracy' not in col and 'task_status' not in col:
            df_dict['taskswitch'] = df_dict['taskswitch'].drop(columns=[col])
# Still 8 columns - taking the mean between relative and parity ==> 4 columns

# we only use the switching costs.
        if 'switching' not in col and 'task_status' not in col:
            df_dict['taskswitch'] = df_dict['taskswitch'].drop(columns=[col])

for col in df_dict['gonogo'].columns:
    if col != 'participant_id':
        if 'rt' not in col and 'task_status' not in col:
            df_dict['gonogo'] = df_dict['gonogo'].drop(columns=[col])

for col in df_dict['memorability'].columns:
    if col != 'participant_id':
        if 'rt' in col and 'task_status' not in col:
            df_dict['memorability'] = df_dict['memorability'].drop(columns=[col])

for col in df_dict['moteval'].columns:
    if col != 'participant_id':
        if 'rt' in col and 'task_status' not in col:
            df_dict['moteval'] = df_dict['moteval'].drop(columns=[col])
# Memorability 20 columns
# First, drop the rt_std column
for col in df_dict['memorability'].columns:
    if 'std' in col:
        df_dict['memorability'] = df_dict['memorability'].drop(columns=[col])
# Let's check covariance matrix between rt conditions
# rt_names = [rt_name for rt_name in df_dict['memorability'].columns if 'rt_' in rt_name]
# hit_names = [rt_name for rt_name in df_dict['memorability'].columns if 'hit_' in rt_name]
# fa_names = [rt_name for rt_name in df_dict['memorability'].columns if 'fa_' in rt_name]
# print(df_dict['memorability'][rt_names].cov())
# conditions_to_group = ['2', '3', '4', '5']


def group_by_conditions(df, conditions_to_group):
    df['out_mat_rt_short'] = df[[f"out_mat_rt_cond-{elt}" for elt in conditions_to_group]].mean(axis=1)
    df['out_mat_hit_miss_short'] = df[[f"out_mat_hit_miss_sum-{elt}" for elt in conditions_to_group]].mean(axis=1)
    df['out_mat_fa_cr_short'] = df[[f"out_mat_fa_cr_sum-{elt}" for elt in conditions_to_group]].mean(axis=1)


# group_by_conditions(df_dict['memorability'], conditions_to_group)
# for col in df_dict['memorability'].columns:
#     if col != 'participant_id':
#         if "short" not in col and "100" not in col:
#             df_dict['memorability'].drop(columns=[col], inplace=True)

for task_name, df in df_dict.items():
    #df.drop(columns=[col for col in df.columns if '-rt' in col], inplace=True)
    df.drop(columns=[col for col in df.columns if '-fa' in col], inplace=True)

dataframe = pd.DataFrame()
for i, (task_name, df) in enumerate(df_dict.items()):
    if 'total_resp' in df.columns:
        df = df.drop(columns=['total_resp'])
    # Add the task name after each column
    for col in df.columns:
        if 'participant_id' not in col and 'task_status' not in col:
            df = df.rename(columns={col: f'{col}-{task_name}'})
    if i == 0:
        dataframe = df
    else:
        dataframe = pd.merge(dataframe, df, on=['participant_id', 'task_status'], how='outer')

dataframe.to_csv('tmp_merge_acp_all_including_rt.csv')


