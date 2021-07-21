import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from extract_sorted_memory import Results_memory
from cal_stan_accuracy_rt import CalStan_accuracy,CalStan_rt




# Treat data:
def compute_result_exact_answers(row):
    response = row["results_responses"].split(',')
    target = row["results_targetvalue"].split(',')

    return sum(x == y for x, y in zip(response, target))

def delete_uncomplete_participants(dataframe):
    """

    """
    mask = pd.DataFrame(dataframe.participant_id.value_counts() < 2)
    participants_to_delete = mask[mask['participant_id'] == True].index.tolist()
    for id in participants_to_delete:
        dataframe = dataframe[dataframe['participant_id'] != id]
    return dataframe

def extract_id(dataframe,num_count):
    mask = pd.DataFrame(dataframe.participant_id.value_counts() == num_count)
    indices_id = mask[mask['participant_id'] == True].index.tolist()
    return indices_id


def draw_all_distributions(dist_ind,dist_summary,num_dist,num_cond,std_val = 0.05,fname_save='memorability_accuracy.png'):
    # dist_ind is the matrix of (num_observers x num_conditions)
    # dist_summary is the mu (numb_conditions), ci_min, and ci_max
    x = [j+1+std_val*np.random.randn() for j in range(num_cond) for t in range(num_dist)]
    dist_ind = dist_ind.T.flatten()
    x_sum = np.linspace(1,num_cond,num_cond)
    fig = plt.figure(figsize=(5,5))
    axes = fig.add_subplot(1,1,1)
    axes.scatter(x,dist_ind,s=10,c='blue')
    axes.errorbar(x_sum, dist_summary[:,0], yerr = [dist_summary[:,0]-dist_summary[:,1], dist_summary[:,2]-dist_summary[:,0]], capsize=5, fmt='o', markersize=15, ecolor='red', markeredgecolor = "red", color='w')
    axes.set_xticks([1,2,3,4,5])
    axes.set_xticklabels(['2','3','4','5','>100'],fontsize=20)
    axes.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    axes.set_yticklabels(['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=20)
    fig.savefig(fname_save)
    plt.show()


def draw_all_distributions_rt(dist_ind,dist_summary,num_dist,num_cond,std_val = 0.05,fname_save='memorability_rt.png'):
    # dist_ind is the matrix of (num_observers x num_conditions)
    # dist_summary is the mu (numb_conditions), ci_min, and ci_max
    x = [j+1+std_val*np.random.randn() for j in range(num_cond) for t in range(num_dist)]
    dist_ind = dist_ind.T.flatten()
    x_sum = np.linspace(1,num_cond,num_cond)
    fig = plt.figure(figsize=(5,5))
    axes = fig.add_subplot(1,1,1)
    axes.scatter(x,dist_ind,s=10,c='blue')
    axes.errorbar(x_sum, dist_summary[:,0], yerr = [dist_summary[:,0]-dist_summary[:,1], dist_summary[:,2]-dist_summary[:,0]], capsize=5, fmt='o', markersize=15, ecolor='red', markeredgecolor = "red", color='w')
    axes.set_xticks([1,2,3,4,5])
    axes.set_xticklabels(['2','3','4','5','>100'],fontsize=20)
    axes.set_yticks([0,400,800,1200])
    axes.set_yticklabels(['0','400','800','1200'],fontsize=20)
    fig.savefig(fname_save)
    plt.show()



def extract_mu_ci_from_summary_accuracy(dataframe,ind_cond):
    outs = np.zeros((len(ind_cond),3)) #3 means the mu, ci_min, and ci_max
    for t,ind in enumerate(ind_cond):
        outs[t,0] = dataframe[ind].mu_theta
        outs[t,1] = dataframe[ind].ci_min
        outs[t,2] = dataframe[ind].ci_max
    return outs

def extract_mu_ci_from_summary_rt(dataframe,ind_cond):
    outs = np.zeros((len(ind_cond),3)) #3 means the mu, ci_min, and ci_max
    for t,ind in enumerate(ind_cond):
        outs[t,0] = dataframe[ind].mu_rt
        outs[t,1] = dataframe[ind].ci_min
        outs[t,2] = dataframe[ind].ci_max
    return outs
if __name__=='__main__':

    csv_path_1 = "outputs/memorability/memorability_1.csv"
    dataframe_1 = pd.read_csv(csv_path_1)
    dataframe_1 = delete_uncomplete_participants(dataframe_1)

    csv_path_2 = "outputs/memorability/memorability_2.csv"
    dataframe_2 = pd.read_csv(csv_path_2)
    dataframe_2 = delete_uncomplete_participants(dataframe_2)

    dataframe = pd.concat([dataframe_1, dataframe_2],axis = 0)
    indices_id = extract_id(dataframe,num_count=4)
    
    sum_observers = []
    for ob in indices_id:
        print(ob)
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        tmp_results = Results_memory(tmp_df)
        sum_observers.append(np.concatenate((tmp_results.out_mat_hit_miss_sum,tmp_results.out_mat_fa_cr_sum,
                                            tmp_results.out_mat_rt_cond,tmp_results.out_mat_rt_cond_std)))
    
    sum_observers = pd.DataFrame(sum_observers)
    #for save summary data
    #tmp = sum_observers/32.
    #tmp.loc[:,10:20] = tmp.loc[:,10:20]*32 
    #tmp.to_csv('sumdata_memorability.csv',header=False, index=False)
    sum_observers['total_resp'] = sum_observers.apply(lambda row: 32, axis=1)
    

    #import pdb;pdb.set_trace()
    class_stan_accuracy = [CalStan_accuracy(sum_observers,ind_corr_resp=n) for n in range(10)]
    class_stan_rt = [CalStan_rt(sum_observers,ind_rt=10+n,max_rt=1400) for n in range(5)]
    

    #for hr data
    dist_ind = sum_observers.iloc[0:len(sum_observers),0:5].values/32.
    dist_summary = extract_mu_ci_from_summary_accuracy(class_stan_accuracy,[0,1,2,3,4])
    draw_all_distributions(dist_ind,dist_summary,len(sum_observers),num_cond=5,std_val = 0.05,fname_save='memorability_hr.png')
    
    #for far data
    dist_ind = sum_observers.iloc[0:len(sum_observers),5:10].values/32.
    dist_summary = extract_mu_ci_from_summary_accuracy(class_stan_accuracy,[5,6,7,8,9])
    draw_all_distributions(dist_ind,dist_summary,len(sum_observers),num_cond=5,std_val = 0.05,fname_save='memorability_far.png')
    
    dist_ind = sum_observers.iloc[0:len(sum_observers),10:15].values
    dist_summary = extract_mu_ci_from_summary_rt(class_stan_rt,[0,1,2,3,4])
    draw_all_distributions_rt(dist_ind,dist_summary,len(sum_observers),num_cond=5,std_val = 0.05,fname_save='memorability_rt.png')

    import pdb;pdb.set_trace()
    print('finished')