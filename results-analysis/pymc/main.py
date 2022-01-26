import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pymc3 as pm

# %config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

import logging

logger = logging.getLogger("pymc3")
logger.propagate = False
from scipy import stats
from data import get_data, convert_to_global_task
from model import PooledModel, UnpooledModel

if __name__ == '__main__':
    tasks = get_data()
    for condition, (data, condition_list) in tasks.items():
        model_zpdes = UnpooledModel(data[data['condition'] == 'zpdes'], name=f'{condition}_zpdes',
                                  stim_cond_list=condition_list)
        model_zpdes.run()
        model_baseline = UnpooledModel(data[data['condition'] == 'baseline'], name=f'{condition}_baseline',
                                     stim_cond_list=condition_list)
        model_baseline.run()
        model_zpdes.compare_traces(model_baseline.traces, 'difference_of_means')
        model_zpdes.compare_traces(model_baseline.traces, 'pre_test_theta')
        model_zpdes.compare_traces(model_baseline.traces, 'post_test_theta')

    # model_zpdes = PooledModel(tasks['mot'][tasks['mot']['condition'] == 'zpdes'], name='zpdes_mot')
    # model_baseline = PooledModel(tasks['mot'][tasks['mot']['condition'] == 'baseline'], name='baseline')
    # model_zpdes.run()
    # model_baseline.run()
    # model_zpdes = PooledModel(tasks['mot'],  name='zpdes_mot', traces_path='zpdes_mot')
    # model_baseline = PooledModel(tasks['mot'],  name='baseline', traces_path='baseline')
    # model_zpdes.plot_CV_figures()
    # model_zpdes.compare_traces(model_baseline.traces, 'difference_of_means')
    # model_zpdes.compare_traces(model_baseline.traces, 'pre_test_theta')
    # model_zpdes.compare_traces(model_baseline.traces, 'post_test_theta')
