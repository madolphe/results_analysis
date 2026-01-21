import pandas as pd
import numpy as np
df1 = pd.read_csv("df_before_pca.csv").drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
df2 = pd.read_csv("tmp_merge_acp_all_including_rt.csv").drop(columns=['Unnamed: 0'])


def check_identical_rows(df1, df2, key_columns):
    # Merge the DataFrames on key_columns
    merged_df = pd.merge(df1, df2, on=key_columns, suffixes=('_df1', '_df2'))

    # Iterate through each row and compare the values of non-key columns
    for _, row in merged_df.iterrows():
        for col in df1.columns:
            if col not in key_columns:
                if np.abs(row[col + '_df1'] - row[col + '_df2']) > 0.000001:
                    return False, col

    return True, None

# Example usage
result, col = check_identical_rows(df1, df2, ['participant_id', 'task_status'])
print("DataFrames are identical for each participant_id and task_status pair:", result, col)

if __name__ == '__main__':
    print('a')