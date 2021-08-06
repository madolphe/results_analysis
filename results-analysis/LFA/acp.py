import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Delete FA for memorability
# Delete RT for all
# Plot RDM

if __name__ == '__main__':
    pca = PCA(0.99)
    dataframe = pd.read_csv('tmp_merge_acp_all.csv')
    dataframe = dataframe.dropna()
    # dataframe = dataframe[dataframe.columns[:-20]]
    normalized_dataframe = (dataframe - dataframe.mean()) / dataframe.std()
    cov_matrix = normalized_dataframe.cov()
    pca.fit(dataframe.values)
    plt.scatter([i for i in range(1, len(pca.explained_variance_ratio_) + 1)], np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.xticks(np.arange(1, len(pca.explained_variance_ratio_) + 1),
               [f"PC{i}" for i in range(1, len(pca.explained_variance_ratio_) + 1)])
    plt.ylabel('Explained variance')
    plt.show()
    results_loading = pd.DataFrame(pca.components_, columns=dataframe.columns).T
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
