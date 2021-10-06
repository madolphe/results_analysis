import pandas as pd


def transform_accuracy_to_total(dataframe, col_name, nb_ans):
    dataframe[col_name] *= nb_ans


def export_to_2_csv(dataframe, name):
    dataframe[dataframe['task_status'] == 'POST_TEST'].to_csv(f"{name}_post.csv")
    dataframe[dataframe['task_status'] == 'PRE_TEST'].to_csv(f"{name}_pre.csv")


if __name__ == '__main__':
    df = pd.read_csv('../outputs/enumeration/enumeration_lfa.csv')
    cols = ["5-accuracy", "6-accuracy", "7-accuracy", "8-accuracy", "9-accuracy"]
    dict_task = {"enumeration": {"col_name": ["5-accuracy", "6-accuracy", "7-accuracy", "8-accuracy", "9-accuracy"],
                                 "nb_ans": [20] * 5},
                 "gonogo": {"col_name": [],
                            "nb_ans": []}}
    for col in cols:
        transform_accuracy_to_total(df, col, 20)
    export_to_2_csv(df, "enumeration")
