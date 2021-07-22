import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import *
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
    draw_all_distributions(dist_ind,dist_summary,len(sum_observers),num_cond=5,std_val = 0.05,
                                list_xlim=[0.75,5.25],list_ylim=[0,1],
                                list_set_xticklabels=['5','6','7','8','9'],list_set_xticks=[1,2,3,4,5],
                                list_set_yticklabels=['0.0','0.2','0.4','0.6','0.8','1.0'],list_set_yticks=[0,0.2,0.4,0.6,0.8,1.0],
                                fname_save='../outputs/enumeration/enumeration_accuracy.png')
    print('finished')