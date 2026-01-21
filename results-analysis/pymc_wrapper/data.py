import copy

import pandas as pd
import os


def change_accuracy_for_correct_column(column_name: str):
    return column_name.replace('accuracy', 'correct')


def get_data(path='../../outputs/v1_ubx/'):
    # # ENUMERATION  # #
    df_enum_csv = pd.read_csv(os.path.join(path, "enumeration_lfa_v1.csv"))
    df_enum = copy.deepcopy(df_enum_csv)
    df_enum = df_enum.rename(change_accuracy_for_correct_column, axis='columns')
    df_enum[[col for col in df_enum.columns if 'correct' in col]] = df_enum[[col for col in df_enum.columns if
                                                                             'correct' in col]] * 20
    enum_cdt = ['5', '6', '7', '8', '9']
    for cdt in enum_cdt:
        df_enum[cdt + '-nb'] = 20
    # df_enum['total_resp'] = 20
    df_enum['total-task-correct'] = convert_to_global_task(df_enum, [col + '-correct' for col in enum_cdt])
    df_enum['total-task-nb'] = 20 * len(enum_cdt)
    df_enum = df_enum.groupby('participant_id').filter(lambda x: len(x) > 1)
    enum_cdt.append('total-task')

    # # WORKING MEMORY # #
    df_wm = pd.read_csv(os.path.join(path, "workingmemory_lfa.csv"))
    wm_cdt = ['4', '5', '6', '7', '8']
    df_wm = df_wm.rename(change_accuracy_for_correct_column, axis='columns')
    df_wm[[col for col in df_wm.columns if 'correct' in col]] = df_wm[[col for col in df_wm.columns if
                                                                       'correct' in col]] * 12
    # df_wm['total_resp'] = 12
    for cdt in wm_cdt:
        df_wm[cdt + '-nb'] = 12
    df_wm['total-task-correct'] = convert_to_global_task(df_wm, [col + '-correct' for col in wm_cdt])
    df_wm['total-task-nb'] = 12 * len(wm_cdt)
    wm_cdt.append('total-task')

    # # MOT # #
    df_mot = pd.read_csv(os.path.join(path, "moteval_lfa.csv"))
    mot_cdt = ['1', '4', '8']
    df_mot = df_mot.rename(change_accuracy_for_correct_column, axis='columns')
    df_mot[[col for col in df_mot.columns if 'correct' in col]] = df_mot[[col for col in df_mot.columns if
                                                                          'correct' in col]] * 15 * 5
    for cdt in mot_cdt:
        df_mot[cdt + '-nb'] = 15 * 5
    # df_mot['total_resp'] = 15 * 5
    df_mot['total-task-correct'] = convert_to_global_task(df_mot, [col + '-correct' for col in mot_cdt])
    df_mot['total-task-nb'] = 45 * 5
    mot_cdt.append('total-task')

    # # TASK-SWITCH # #
    df_tsw = pd.read_csv(os.path.join(path, "taskswitch_lfa.csv"))
    # Condition to check ==> accuracy in parity task or in relative task:
    df_tsw['parity-correct'] = df_tsw['parity-switch-correct'] + df_tsw['parity-unswitch-correct']
    df_tsw['parity-nb'] = df_tsw['parity-switch-nb'] + df_tsw['parity-unswitch-nb']
    df_tsw['relative-correct'] = df_tsw['relative-switch-correct'] + df_tsw['relative-unswitch-correct']
    df_tsw['relative-nb'] = df_tsw['relative-switch-nb'] + df_tsw['relative-unswitch-nb']
    # Other condition to check ==> accuracy in switch VS unswitch condition
    df_tsw['switch-correct'] = df_tsw['parity-switch-correct'] + df_tsw['relative-switch-correct']
    df_tsw['switch-nb'] = df_tsw['parity-switch-nb'] + df_tsw['relative-switch-nb']
    df_tsw['unswitch-correct'] = df_tsw['parity-unswitch-correct'] + df_tsw['relative-unswitch-correct']
    df_tsw['unswitch-nb'] = df_tsw['parity-unswitch-nb'] + df_tsw['relative-unswitch-nb']
    df_tsw = df_tsw.drop(
        columns=['parity-switch-correct', 'parity-unswitch-correct', 'parity-switch-nb', 'parity-unswitch-nb',
                 'relative-switch-correct', 'relative-unswitch-nb', 'relative-unswitch-correct', 'relative-switch-nb'])
    tsw_cdt = ['parity', 'relative', 'switch', 'unswitch']
    # Switch and unswitch correct contains whether the participant answered relative/parity task:
    df_tsw['total-task-correct'] = convert_to_global_task(df_tsw, ['switch-correct', 'unswitch-correct'])
    df_tsw['total-task-nb'] = df_tsw['switch-nb'] + df_tsw['unswitch-nb']
    tsw_cdt.append('total-task')

    # # LOADBLINDNESS # #
    df_lb = pd.read_csv(os.path.join(path, "loadblindness_lfa.csv"))
    lb_cdt = ['near', 'far']
    df_lb[[col for col in df_lb.columns if 'accuracy' in col]] = df_lb[[col for col in df_lb.columns if
                                                                        'accuracy' in col]] * 20
    df_lb = df_lb.rename(columns={'accuracy_near': 'near-correct', 'accuracy_far': 'far-correct'})
    # df_lb['total_resp'] = 20
    for cdt in lb_cdt:
        df_lb[cdt + '-nb'] = 20
    df_lb['total-task-correct'] = convert_to_global_task(df_lb, [col + '-correct' for col in lb_cdt])
    df_lb['total-task-nb'] = 40
    lb_cdt.append('total-task')

    # # GO / NO GO # #
    df_go = pd.read_csv(os.path.join(path, "gonogo_lfa.csv"))
    df_go = df_go.rename(columns={'HR-accuracy': 'GO-accuracy', 'FAR-accuracy': 'NOGO-accuracy', 'HR-rt': 'GO-rt'})
    go_cdt = ['GO', 'NOGO']
    df_go = df_go.rename(change_accuracy_for_correct_column, axis='columns')
    df_go[[col for col in df_go.columns if 'correct' in col]] = df_go[[col for col in df_go.columns if
                                                                       'correct' in col]] * 18
    for cdt in go_cdt:
        df_go[cdt + '-nb'] = 20
    df_go['total-task-correct'] = df_go['GO-correct'] + (18 - df_go['NOGO-correct'])
    df_go['total-task-nb'] = 36
    go_cdt.append('total-task')

    # # MEMORABILITY # #
    df_memora = pd.read_csv(os.path.join(path, "memorability_lfa.csv"))
    memora_cdt = ['2', '3', '4', '5', '100']
    for col in df_memora.columns:
        if 'hit-miss' in col:
            df_memora = df_memora.rename(columns={col: col.replace('hit-miss', 'correct')})
        if 'fa' in col:
            df_memora = df_memora.drop(columns=[col])
    for col in memora_cdt:
        df_memora[col+'-nb'] = 16
    df_memora['total-task-correct'] = convert_to_global_task(df_memora, [col + '-correct' for col in memora_cdt])
    df_memora['total-task-nb'] = 80
    df_memora.dropna(inplace=True)
    df_memora = df_memora.groupby('participant_id').filter(lambda x: len(x) > 1)
    memora_cdt.append('total-task')

    tasks = {'enumeration': (df_enum, enum_cdt),
             'working_memory': (df_wm, wm_cdt),
             'gonogo': (df_go, go_cdt),
             'loadblindness': (df_lb, lb_cdt),
             'mot': (df_mot, mot_cdt),
             'memorability': (df_memora, memora_cdt),
             'taskswitch': (df_tsw, tsw_cdt)}
    # tasks_rt = {
    #     'gonogo': (df_go, ['GO']),
    #     'taskswitch': (df_tsw, ['parity-switching-cost', 'relative-switching-cost']),
    #     'mot': (df_mot, ['1', '4', '8']),
    #     'memorability': (df_memora, ['2', '3', '4', '5', '100'])
    # }
    tasks_rt = {
        'taskswitch': (df_tsw, ['parity-switching-cost', 'relative-switching-cost']),
    }
    # tasks_rt = {
    #     'gonogo': (df_go, ['GO'])
    # }
    # tasks = {'memorability': (df_memora, memora_cdt)}
    # tasks = {'enumeration': (df_enum, ['total-task'])}
    # tasks = {'taskswitch': (df_tsw, ['parity-switching-cost'])}
    return tasks_rt


def convert_to_global_task(task, conditions):
    return task[conditions].sum(axis=1)
