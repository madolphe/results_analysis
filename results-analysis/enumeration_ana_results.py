import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cal_stan_accuracy_rt import CalStan_accuracy, CalStan_rt

# Treat data:
def compute_result_exact_answers(row):
    response = row["results_responses"].split(',')
    target = row["results_targetvalue"].split(',')
    return sum(x == y for x, y in zip(response, target))


def compute_mean_per_row(row):
    return_value = np.array(row['results_rt'].split(','), dtype=np.int32)
    return np.mean(return_value)


def compute_std_per_row(row):
    return_value = np.array(row['results_rt'].split(','), dtype=np.int32)
    return np.std(return_value)


def compute_result_exact_answers_list(row):
    response = row["results_responses"].split(',')
    target = row["results_targetvalue"].split(',')
    out =  [1 if x == y else 0 for x, y in zip(response, target)]
    return np.array(out)

def compute_numbercond(row,ind_cond):
    """
        From the row of results, return the list of resuluts_correct according to results_num_stim condition 
    """
    results_responses = list(row["result_correct"])
    results_targetvalue = [int(t) for t in row["results_targetvalue"].split(',')]
    out = [results_responses[ind] for ind,t in enumerate(results_targetvalue) if t==ind_cond]
    return np.array(out)

def compute_sum_to_row(row,column):
    return np.sum(row[column])

def extract_id(dataframe,num_count):
    mask = pd.DataFrame(dataframe.participant_id.value_counts() == num_count)
    indices_id = mask[mask['participant_id'] == True].index.tolist()
    return indices_id

def extract_mu_ci_from_summary_accuracy(dataframe,ind_cond):
    out = np.zeros((len(ind_cond),3)) #3 means the mu, ci_min, and ci_max
    for t,ind in enumerate(ind_cond):
        out[t,0] = dataframe[ind].mu_theta
        out[t,1] = dataframe[ind].ci_min
        out[t,2] = dataframe[ind].ci_max
    return out

def draw_all_distributions(dist_ind,dist_summary,num_dist,num_cond,std_val = 0.05,fname_save='workingmemory_accuracy.png'):
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
    #axes.set_xlim([0.5,2.5])
    axes.set_xticklabels(['5','6','7','8','9'],fontsize=20)
    axes.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    axes.set_yticklabels(['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=20)
    fig.savefig(fname_save)
    plt.show()

if __name__ == '__main__':
    csv_path = "../outputs/enumeration/enumeration.csv"
    dataframe = pd.read_csv(csv_path)
    dataframe['result_response_exact'] = dataframe.apply(compute_result_exact_answers, axis=1)

    dataframe['mean_rt_session'] = dataframe.apply(compute_mean_per_row, axis=1)
    dataframe['std_rt_session'] = dataframe.apply(compute_std_per_row, axis=1)


    # Reliability of measurement
    pre_response_exact = dataframe[dataframe['task_status'] == "PRE_TEST"]['result_response_exact'].values
    post_response_exact = dataframe[dataframe['task_status'] == "POST_TEST"]['result_response_exact'].values

    pearson_coeff = np.corrcoef(pre_response_exact, post_response_exact)[1, 0]**2

    plt.scatter(pre_response_exact, post_response_exact)
    plt.title(f"Pearson coefficient: {pearson_coeff}")

    # Mean and SD reaction time plots and values:
    fig, axs = plt.subplots(1, len(dataframe.task_status.unique()), figsize=(10,  5), sharey=False)
    boxplot = dataframe.boxplot(column=['mean_rt_session', 'result_response_exact'], by=['task_status'], layout=(2, 1), ax=axs)
    #plt.show()


    #from here written by mswym
    #condition extraction
    dataframe['result_correct'] = dataframe.apply(compute_result_exact_answers_list, axis=1)
    number_condition = [5,6,7,8,9]
    for t in number_condition:
        dataframe[str(t)] = dataframe.apply(lambda row: compute_numbercond(row, t), axis=1)
        dataframe['sum'+str(t)] = dataframe.apply(lambda row: compute_sum_to_row(row, str(t)), axis=1)

    #extract observer index information
    indices_id = extract_id(dataframe,num_count=2)
    
    #sumirize two days experiments
    sum_observers = []
    for ob in indices_id:
        print(ob)
        tmp_df = dataframe.groupby(["participant_id"]).get_group(ob)
        sum_observers.append([np.sum(tmp_df.sum5),np.sum(tmp_df.sum6),np.sum(tmp_df.sum7),np.sum(tmp_df.sum8),np.sum(tmp_df.sum9)])
        
    sum_observers = pd.DataFrame(sum_observers)
    import pdb;pdb.set_trace()
    #for save summary data
    tmp = sum_observers/40.
    tmp.to_csv('../outputs/enumeration/sumdata_enumeration.csv',header=False, index=False)
    sum_observers['total_resp'] = sum_observers.apply(lambda row: 40, axis=1) #two days task

    #calculate the mean distribution and the credible interval
    class_stan_accuracy = [CalStan_accuracy(sum_observers,ind_corr_resp=n) for n in range(5)]

    #draw figures
    #for accuracy data
    dist_ind = sum_observers.iloc[0:len(sum_observers),0:5].values/40.
    dist_summary = extract_mu_ci_from_summary_accuracy(class_stan_accuracy,[0,1,2,3,4])
    draw_all_distributions(dist_ind,dist_summary,len(sum_observers),num_cond=5,std_val = 0.05,fname_save='../outputs/enumeration/enumeration_accuracy.png')
    print('finished')