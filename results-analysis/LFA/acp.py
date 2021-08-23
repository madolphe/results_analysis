import pandas as pd
from sklearn.decomposition import PCA, FastICA,DictionaryLearning
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import norm

# Delete FA for memorability
# Delete RT for all
# Plot RDM

def inv_norm_transform(dataframe):
    #first 0 and 1 number is replaced according to the total task number to avoid infinity. See also Macmillan, N. A., & Kaplan, H. L. (1985).
    task_names = ['moteval', 'loadblindness', 'memorability', 'enumeration', 'workingmemory']
    num_totals = [15.,20.,16.,20.,12.]
    for col in dataframe.columns:
        for task,num in zip(task_names,num_totals):
            if task in col:
                dataframe[col] = dataframe[col].mask(dataframe[col]==1,1-(1/(2*num)))
                dataframe[col] = dataframe[col].mask(dataframe[col]==0,1/(2*num))
        #invert normal transformation is applyed only for the accuracy data, not for rt. For rt data, I apply log transform.
        if 'rt' not in col:
            dataframe[col] = dataframe[col].apply(norm.ppf)
        else:
            dataframe[col] = dataframe[col].apply(np.log)
    return dataframe

if __name__ == '__main__':
    pca = PCA(10)
    dataframe = pd.read_csv('tmp_merge_acp_all_plus_rt.csv')
    dataframe = dataframe.fillna(dataframe.mean())
    dataframe.drop(columns=['Unnamed: 0', "task_status", 'participant_id'], inplace=True)
    # dataframe = dataframe[dataframe.columns[:-20]]
    dataframe = inv_norm_transform(dataframe)
    normalized_dataframe = (dataframe - dataframe.mean()) / dataframe.std()
    cov_matrix = normalized_dataframe.cov()
    pca.fit(normalized_dataframe.values)
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
    #sns.heatmap(results_loading.abs().values)
    sns.heatmap(results_loading.values,cmap="PuOr",center=0)
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


    ica = FastICA(10)
    ica.fit(normalized_dataframe.values)
    Uica = ica.mixing_ 
    Aica = ica.transform(normalized_dataframe.values).T
    plt.figure()
    sns.heatmap(Uica,cmap="PuOr",center=0)
    plt.savefig('heatmap_loadings_ica.png')
    plt.close()

    sparse = DictionaryLearning(10)
    sparse.fit(normalized_dataframe.values)
    Usc = sparse.components_ .T
    Asc = sparse.transform(normalized_dataframe.values).T
    plt.figure()
    sns.heatmap(Usc,cmap="PuOr",center=0)
    plt.savefig('heatmap_loadings_sparse.png')
    plt.close()
    import pdb;pdb.set_trace()
    a =1
