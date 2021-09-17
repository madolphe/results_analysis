import pandas as pd
from sklearn.decomposition import PCA, FastICA,DictionaryLearning
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.stats import norm
from sklearn.manifold import TSNE
import numpy.matlib
import copy
from sklearn.model_selection import KFold

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
        #I changed the log transformation because the switching cost can be minus
        #else:
            #dataframe[col] = dataframe[col].apply(np.log)
    return dataframe


def cal_draw_pca(dataframe,train):
    data_extract = copy.deepcopy(dataframe.values[train,:])
    pca.fit(data_extract)
    plt.bar([i for i in range(1, len(pca.explained_variance_ratio_) + 1)], pca.explained_variance_ratio_)
    plt.plot([i for i in range(1, len(pca.explained_variance_ratio_) + 1)], pca.explained_variance_ratio_,'ro-',c='red')
    plt.xlabel('Number of components')
    plt.xticks(np.arange(1, len(pca.explained_variance_ratio_) + 1),
               [f"{i}" for i in range(1, len(pca.explained_variance_ratio_) + 1)])
    plt.ylabel('Explained variance')
    plt.savefig('Cumulative explained variance')
    plt.close()

    print('cumulated percentage is...')
    print(np.cumsum(pca.explained_variance_ratio_))

    results_loading = pd.DataFrame(pca.components_, columns=dataframe.columns).T
    results_loading.to_csv('loadings.csv')


    linked = linkage(results_loading.values, 'single')
    labelList = range(1, 27)

    plt.figure(figsize=(20, 15))
    tmp_fig = dendrogram(linked,
               orientation='left',
               labels=results_loading.T.columns,
               distance_sort='descending',
               show_leaf_counts=True,
               leaf_font_size=20.,
               color_threshold=0)
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig('dendrogram.png')
    #index_dendro = np.matlib.repmat(tmp_fig['leaves'],num_components,1).T
    index_dendro = tmp_fig['leaves']
    plt.close()
    #pca.transform(X.T).T

    plt.figure()
    results_loading_pca = results_loading.values
    results_loading_pca = results_loading_pca[index_dendro]
    axes = sns.heatmap(results_loading_pca,cmap="PuOr",center=0)
    axes.invert_yaxis()
    plt.savefig('heatmap_loadings.png')
    plt.close()

    return pca

def validate_model(dataframe,test,model):
    #model is trained one using pca.fit or ica.fit
    data_extract = copy.deepcopy(dataframe.values[test,:])
    codes = model.transform(data_extract)
    data_est = model.inverse_transform(codes)
    rmse = np.sqrt(np.mean((data_extract.flatten()-data_est.flatten())**2))
    coeff = np.corrcoef(data_extract.flatten(),data_est.flatten())
    coeff = coeff[0,1]
    return rmse, coeff


if __name__ == '__main__':
    num_components = 23
    pca = PCA(num_components)
    dataframe = pd.read_csv('tmp_merge_acp_all_including_rt.csv')
    dataframe = dataframe.fillna(dataframe.mean())
    dataframe.drop(columns=['Unnamed: 0', "task_status", 'participant_id'], inplace=True)
    # dataframe = dataframe[dataframe.columns[:-20]]
    dataframe = inv_norm_transform(dataframe)
    normalized_dataframe = (dataframe - dataframe.mean()) / dataframe.std()
    normalized_dataframe_shuffle = normalized_dataframe.sample(frac=1, random_state=0)
    cov_matrix = normalized_dataframe.cov()

    kf = KFold(n_splits=10)
    pca_coeff = []
    pca_rmse = []
    pca_eigens = []
    for train, test in kf.split(normalized_dataframe_shuffle):
        pca = cal_draw_pca(normalized_dataframe_shuffle,train)
        pca_eigens.append(pca.explained_variance_ratio_)
        rmse,coeff = validate_model(normalized_dataframe_shuffle,test,pca)
        pca_rmse.append(rmse)
        pca_coeff.append(coeff)

    
    #draw the eigen values.
    mean_pca_eigens = np.mean(np.array(pca_eigens),0)
    error_pca_eigen = np.std(np.array(pca_eigens),0)
    
    error_bar_set = dict(lw = 1, capthick = 1, capsize = 10, c='red')
    fig = plt.figure(figsize=(8, 6))
    axes = fig.add_subplot(1, 1, 1)
    #axes.grid()
    axes.bar([i for i in range(1, len(mean_pca_eigens) + 1)], mean_pca_eigens,yerr=error_pca_eigen,error_kw=error_bar_set)
    axes.plot([i for i in range(1, len(mean_pca_eigens) + 1)], mean_pca_eigens,'-',c='red')
    axes.set_ylim([0,0.35])
    axes.set_xticks([i for i in range(1, len(mean_pca_eigens) + 1)])
    axes.set_xticklabels([i for i in range(1, len(mean_pca_eigens) + 1)],fontsize=12)
    axes.set_yticks([0.0,0.1,0.2,0.3])
    axes.set_yticklabels(['0.0','0.1','0.2','0.3'], fontsize=12)
    fig.savefig('explained_variance.png')
    plt.close()    
    print('finished')

'''
    sparse = DictionaryLearning(num_components)
    sparse.fit(normalized_dataframe.values)
    Usc = sparse.components_ .T
    Asc = sparse.transform(normalized_dataframe.values).T

    linked = linkage(Usc, 'single')
    plt.figure(figsize=(20, 15))
    tmp_fig = dendrogram(linked,
               orientation='left',
               labels=results_loading.T.columns,
               distance_sort='descending',
               show_leaf_counts=True,
               leaf_font_size=20.,
               color_threshold=0)
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig('dendrogram_sparse.png')
    index_dendro = tmp_fig['leaves']
    plt.close()

    plt.figure()
    Usc = Usc[index_dendro]
    axes = sns.heatmap(Usc,cmap="PuOr",center=0)
    axes.invert_yaxis()
    plt.savefig('heatmap_loadings_sparse.png')
    plt.close()

    tsne = TSNE(n_components=2, perplexity=5, n_iter=1000)
    #tsne_results = tsne.fit_transform(pca.components_.T)
    tsne_results = tsne.fit_transform(normalized_dataframe.values.T)
    labels = [0,1,1,1,2,2,2,2,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6]
    sns.scatterplot(x=tsne_results[:,0],y=tsne_results[:,1],palette=sns.color_palette("hls",7),hue=labels)
    import pdb;pdb.set_trace()
'''