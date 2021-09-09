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
    return dataframe


def cal_draw_pca(dataframe,train,ind):
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
               leaf_font_size=26.,
               color_threshold=0,
               above_threshold_color = 'b')
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig('pca/dendrogram_'+str(ind)+'.png')
    #index_dendro = np.matlib.repmat(tmp_fig['leaves'],num_components,1).T
    index_dendro = tmp_fig['leaves']
    plt.close()
    #pca.transform(X.T).T

    plt.figure()
    results_loading_pca = results_loading.values
    results_loading_pca = results_loading_pca[index_dendro]
    axes = sns.heatmap(results_loading_pca,cmap="PuOr",center=0)
    axes.invert_yaxis()
    plt.savefig('pca/heatmap_loadings_'+str(ind)+'.png')
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

def cal_draw_ica(dataframe,train,ind):
    data_extract = copy.deepcopy(dataframe.values[train,:])
    ica = FastICA(num_components,max_iter=200000)
    ica.fit(data_extract)
    Uica = ica.mixing_ 
    Aica = ica.transform(data_extract).T

    linked = linkage(Uica, 'single')
    plt.figure(figsize=(20, 15))
    tmp_fig = dendrogram(linked,
               orientation='left',
               labels=dataframe.columns,
               distance_sort='descending',
               show_leaf_counts=True,
               leaf_font_size=26.,
               color_threshold=0,
               above_threshold_color = 'b')
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig('ica/dendrogram_ica_'+str(ind)+'.png')
    index_dendro = tmp_fig['leaves']
    plt.close()

    plt.figure()
    Uica = Uica[index_dendro]
    axes = sns.heatmap(Uica,cmap="PuOr",center=0)
    axes.invert_yaxis()
    plt.savefig('ica/heatmap_loadings_ica_'+str(ind)+'.png')
    plt.close()

    return ica



if __name__ == '__main__':
    num_components = 5
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
    ica_coeff = []
    ica_rmse = []
    pca_explained = []
    pca_components = []
    i = 0
    for train, test in kf.split(normalized_dataframe_shuffle):
        i += 1
        pca = cal_draw_pca(normalized_dataframe_shuffle,train,i)
        rmse_pca,coeff_pca = validate_model(normalized_dataframe_shuffle,test,pca)
        pca_rmse.append(rmse_pca)
        pca_coeff.append(coeff_pca)
        pca_components.append(pca.components_)
        pca_explained.append((np.cumsum(pca.explained_variance_ratio_)[num_components-1]))
        
        ica = cal_draw_ica(normalized_dataframe_shuffle,train,i)
        rmse_ica,coeff_ica = validate_model(normalized_dataframe_shuffle,test,ica)
        data_extract = copy.deepcopy(normalized_dataframe_shuffle.values[test,:])
        codes = ica.transform(data_extract)
        data_est = ica.inverse_transform(codes)
        rmse_ica = np.sqrt(np.mean((data_extract.flatten()-data_est.flatten())**2))
        coeff_ica = np.corrcoef(data_extract.flatten(),data_est.flatten())
        coeff_ica = coeff_ica[0,1]
        ica_rmse.append(rmse_ica)
        ica_coeff.append(coeff_ica)

    

    #draw validation data
    print('standard deviation of cross validation std is')
    print(np.std(pca_rmse))
    print('standard error of cross validation std is')
    print(np.std(pca_rmse)/np.sqrt(len(pca_rmse)))

    mu_coeff = np.array([np.mean(pca_coeff),np.mean(ica_coeff)])
    error_coeff = np.array([np.std(pca_coeff)/np.sqrt(len(pca_coeff)),np.std(ica_rmse)/np.sqrt(len(pca_coeff))])
    error_bar_set = dict(lw = 1, capthick = 1, capsize = 20)

    fig = plt.figure(figsize=(5, 5))
    axes = fig.add_subplot(1, 1, 1)
    axes.grid()
    axes.bar(['pca','ica'],mu_coeff,yerr=error_coeff, error_kw=error_bar_set,zorder=5)
    axes.set_ylim([0,1])
    #axes.set_xticks(list_set_xticks)
    axes.set_xticklabels(['PCA','ICA'], fontsize=20)
    axes.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    axes.set_yticklabels(['0.0','0.2','0.4','0.6','0.8','1.0'], fontsize=20)
    fig.savefig('cross_validation.png')
    plt.close()



    #draw summary data
    pca_components = np.mean(pca_components,0)
    results_loading = pd.DataFrame(np.reshape(pca_components,[num_components,normalized_dataframe.shape[1]]), columns=dataframe.columns).T
    results_loading.to_csv('loading_summary.csv')


    linked = linkage(results_loading.values, 'single')
    labelList = range(1, 27)

    plt.figure(figsize=(20, 15))
    tmp_fig = dendrogram(linked,
               orientation='left',
               labels=results_loading.T.columns,
               distance_sort='descending',
               show_leaf_counts=True,
               leaf_font_size=26.,
               color_threshold=0,
               above_threshold_color = 'b')
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig('dendrogram_pcasummary.png')
    #index_dendro = np.matlib.repmat(tmp_fig['leaves'],num_components,1).T
    index_dendro = tmp_fig['leaves']
    plt.close()
    #pca.transform(X.T).T

    plt.figure()
    results_loading_pca = results_loading.values
    results_loading_pca = results_loading_pca[index_dendro]
    axes = sns.heatmap(results_loading_pca,cmap="PuOr",center=0)
    axes.invert_yaxis()
    plt.savefig('heatmap_loadings_pcasummary.png')
    plt.close()
    import pdb;pdb.set_trace()
    print('finished')

