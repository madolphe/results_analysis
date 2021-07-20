import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cal_stan_accuracy_rt import CalStan_accuracy, CalStan_rt

def delete_uncomplete_participants(dataframe):
    """

    """
    mask = pd.DataFrame(dataframe.participant_id.value_counts() < 2)
    participants_to_delete = mask[mask['participant_id'] == True].index.tolist()
    for id in participants_to_delete:
        dataframe = dataframe[dataframe['participant_id'] != id]
    return dataframe

def compute_nearfarcond(row,ind_nearfar):
    """
        From the row of results, return the list of farcondition if elt is min/max in results_targetvalue
        The ind_nearfar 0 means near and 1 means far conditions.
    """
    results_responses = list(row["results_responses_pos"])
    results_targetvalue = list(row["results_target_distance"])
    min_tmp = min(results_targetvalue)
    targind_tmp = []
    targind_tmp = [0 if t==min_tmp else 1  for t in results_targetvalue]
    out = [results_responses[idx] for idx, elt in enumerate(targind_tmp) if elt == ind_nearfar]

    return np.array(out)

def parse_to_int(elt: str) -> int:
    """
        Parse string value into int, if string is null parse to 0
        If null; participant has not pressed the key when expected
    """
    if elt == '':
        return 0
    return int(elt)


def transform_string_to_row(row, column):
    return [int(elt) for elt in row[column].split(',') if elt]


def compute_sum_to_row(row,column):
    return np.sum(row[column])

def extract_id(dataframe,num_count):
    mask = pd.DataFrame(dataframe.participant_id.value_counts() == num_count)
    indices_id = mask[mask['participant_id'] == True].index.tolist()
    return indices_id

def extract_mu_ci_from_summary_accuracy(dataframe,ind_cond):
    outs = np.zeros((len(ind_cond),3)) #3 means the mu, ci_min, and ci_max
    for t,ind in enumerate(ind_cond):
        outs[t,0] = dataframe[ind].mu_theta
        outs[t,1] = dataframe[ind].ci_min
        outs[t,2] = dataframe[ind].ci_max
    return outs

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
    axes.set_xticks([1,2])
    axes.set_xlim([0.5,2.5])
    axes.set_xticklabels(['Near','Far'],fontsize=20)
    axes.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    axes.set_yticklabels(['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=20)
    fig.savefig(fname_save)
    plt.show()

if __name__=='__main__':
    csv_path = "results/loadblindness.csv"
    
    dataframe = pd.read_csv(csv_path)
    dataframe = delete_uncomplete_participants(dataframe)
    #%

    dataframe["results_responses_pos"] = dataframe.apply(lambda row: transform_string_to_row(row, "results_responses_pos"),
                                                     axis=1)
    dataframe["results_target_distance"] = dataframe.apply(lambda row: transform_string_to_row(row, "results_target_distance"),
                                                     axis=1)

    #extract far
    dataframe['far_response'] = dataframe.apply(lambda row: compute_nearfarcond(row, 1), axis=1)
    dataframe['near_response'] = dataframe.apply(lambda row: compute_nearfarcond(row, 0), axis=1)
    dataframe['sum_far'] = dataframe.apply(lambda row: compute_sum_to_row(row, "far_response"), axis=1)
    dataframe['sum_near'] = dataframe.apply(lambda row: compute_sum_to_row(row, "near_response"), axis=1)
    dataframe['total_resp'] = dataframe.apply(lambda row: 20, axis=1)

    #extract observer index information
    indices_id = extract_id(dataframe,num_count=2)
    
    #sumirize two days experiments
    sum_observers = []
    for ob in indices_id:
        print(ob)
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        sum_observers.append([np.sum(tmp_df.sum_near),np.sum(tmp_df.sum_far)])
        
    sum_observers = pd.DataFrame(sum_observers)
    #for save summary data
    #tmp = sum_observers/40.
    #tmp.to_csv('sumdata_loadblindness.csv',header=False, index=False)
    sum_observers['total_resp'] = sum_observers.apply(lambda row: 40, axis=1) #two days task


    #calculate the mean distribution and the credible interval
    class_stan_accuracy = [CalStan_accuracy(sum_observers,ind_corr_resp=n) for n in range(2)]

    #draw figures
    #for accuracy data
    dist_ind = sum_observers.iloc[0:len(sum_observers),0:2].values/40.
    dist_summary = extract_mu_ci_from_summary_accuracy(class_stan_accuracy,[0,1])
    import pdb;pdb.set_trace()
    draw_all_distributions(dist_ind,dist_summary,len(sum_observers),num_cond=2,std_val = 0.05,fname_save='loadblindness_accuracy.png')

    print('finished')