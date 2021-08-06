import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Delete FA for memorability
# Delete RT for all
# Plot RDM

if __name__ == '__main__':
    pca = PCA(6)
    dataframe = pd.read_csv('tmp_merge_acp_all.csv')
    dataframe = dataframe.fillna(dataframe.mean())
    dataframe.drop(columns=['Unnamed: 0', "task_status", 'participant_id'], inplace=True)
    # dataframe = dataframe[dataframe.columns[:-20]]
    normalized_dataframe = (dataframe - dataframe.mean()) / dataframe.std()
    cov_matrix = normalized_dataframe.cov()
    pca.fit(dataframe.values)
    plt.scatter([i for i in range(1, len(pca.explained_variance_ratio_) + 1)], np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.xticks(np.arange(1, len(pca.explained_variance_ratio_) + 1),
               [f"{i}" for i in range(1, len(pca.explained_variance_ratio_) + 1)])
    plt.ylabel('Explained variance')
    plt.savefig('Cumulative explained variance')
    plt.close()

    results_loading = pd.DataFrame(pca.components_, columns=dataframe.columns).T
    results_loading.to_csv('loadings.csv')

    plt.figure()
    sns.heatmap(results_loading.abs().values)
    plt.savefig('heatmap_loadings.png')
    plt.close()

    linked = linkage(results_loading.values, 'single')
    labelList = range(1, 27)

    plt.figure(figsize=(20, 15))
    dendrogram(linked,
               orientation='left',
               labels=results_loading.T.columns,
               distance_sort='descending',
               show_leaf_counts=True)
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig('dendrogram.png')
    plt.close()
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
