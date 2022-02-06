import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pymc3 as pm

# %config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

import logging

logger = logging.getLogger("pymc3")
logger.propagate = False
from data import get_data
from model import PooledModel

if __name__ == '__main__':
    tasks = get_data()
    for condition, (data, condition_list) in tasks.items():
        # model_zpdes = PooledModel(data[data['condition'] == 'zpdes'], name=f'{condition}_zpdes',
        #                           stim_cond_list=condition_list, sample_size=4000)
        # model_zpdes.find_posterior_for_condition()
        model_baseline = PooledModel(data[data['condition'] == 'baseline'], name=f'{condition}_baseline',
                                     stim_cond_list=condition_list, sample_size=4000)
        model_baseline.find_posterior_for_condition()
        # model_baseline.run()
        # model_baseline = PooledModel(data[data['condition'] == 'baseline'], name=f'{condition}_baseline',
        #                              stim_cond_list=condition_list, sample_size=2000)
        # model_baseline.run()
        # model = PooledModelRT(data, name=condition, stim_cond_list=condition_list, sample_size=500)
        # model = PooledModel(data, name=f"{condition}_zpdes", stim_cond_list=condition_list, sample_size=500,
        #                     traces_path='pooled_model/enumeration/enumeration_zpdes_results/enumeration_zpdes-total-task-trace')
        # model.condition = 'total-task'
        # model.get_figures()

# model_zpdes.compare_traces(model_baseline.traces, 'difference_of_means')
# model_zpdes.compare_traces(model_baseline.traces, 'pre_test_theta')
# model_zpdes.compare_traces(model_baseline.traces, 'post_test_theta')
