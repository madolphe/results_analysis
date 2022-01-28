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
        model_zpdes = PooledModel(data[data['condition'] == 'zpdes'], name=f'{condition}_zpdes',
                                  stim_cond_list=condition_list, sample_size=2000)
        model_zpdes.run()
        model_baseline = PooledModel(data[data['condition'] == 'baseline'], name=f'{condition}_baseline',
                                     stim_cond_list=condition_list, sample_size=2000)
        model_baseline.run()
        model_zpdes.compare_traces(model_baseline.traces, 'difference_of_means')
        model_zpdes.compare_traces(model_baseline.traces, 'pre_test_theta')
        model_zpdes.compare_traces(model_baseline.traces, 'post_test_theta')