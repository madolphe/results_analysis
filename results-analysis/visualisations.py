import pandas as pd
# rename each column
# merge all datasets
# Do for each participant a chart w 2 lines (1 PRE / 1 POST)

enumeration_df = pd.read_csv("../outputs/v0_axa/results_v0_axa/enumeration/enumeration_lfa_v1.csv")
gonogo_df = pd.read_csv("../outputs/v0_axa/results_v0_axa/gonogo/gonogo_lfa.csv")
loadblindness_df = pd.read_csv("../outputs/v0_axa/results_v0_axa/loadblindness/loadblindness_lfa.csv")
memorability_df = pd.read_csv("../outputs/v0_axa/results_v0_axa/memorability/memorability_lfa.csv")
moteval_df = pd.read_csv("../outputs/v0_axa/results_v0_axa/moteval/moteval_lfa.csv")
workingmemory_df = pd.read_csv("../outputs/v0_axa/results_v0_axa/workingmemory/workingmemory_lfa.csv")
taskswitch_df = pd.read_csv("../outputs/v0_axa/results_v0_axa/taskswitch/taskswitch_lfa.csv")

if __name__ == '__main__':
    print(enumeration_df)