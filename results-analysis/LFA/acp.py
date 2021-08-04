import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    pca = PCA(n_components=10)
    dataframe = pd.read_csv('tmp_merge_acp.csv')
    dataframe = dataframe.dropna()
    dataframe.drop(columns=['participant_id'], inplace=True)
    normalized_dataframe = (dataframe - dataframe.mean()) / dataframe.std()
    cov_matrix = normalized_dataframe.cov()
    pca.fit(dataframe.values)
    plt.bar(np.arange(0, 10, 1), pca.explained_variance_ratio_)
    plt.show()
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
