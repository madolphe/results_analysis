import json
import enumeration_ana_results
import gonogo_ana_results
import taskswitch_ana_results
import moteval_ana_results
import memorability_ana_results
import loadblindness_ana_results
import workingmemory_ana_results

from pymc.model import PooledModel, PooledModelRTSimulations, PooledModelRTCostSimulations

study = 'v1_ubx'
model_type = 'pooled_model'
with open('config/conditions.JSON', 'r') as f:
    all_conditions = json.load(f)


def run_all(study, accuracy_model=None, RT_model=None, accuracy_model_type=None, RT_model_type=None):
    if accuracy_model:
        enumeration_ana_results.fit_model(study=study, model=accuracy_model,
                                          conditions_to_fit=all_conditions["accuracy"]["enumeration"],
                                          model_type=accuracy_model_type)
        gonogo_ana_results.fit_model(study, model=accuracy_model,
                                     conditions_to_fit=all_conditions["accuracy"]["gonogo"],
                                     model_type=accuracy_model_type)
        taskswitch_ana_results.fit_model(study, model=accuracy_model,
                                         conditions_to_fit=all_conditions["accuracy"]["taskswitch"],
                                         model_type=accuracy_model_type)
        moteval_ana_results.fit_model(study, model=accuracy_model,
                                      conditions_to_fit=all_conditions["accuracy"]["moteval"],
                                      model_type=accuracy_model_type)
        memorability_ana_results.fit_model(study, conditions_to_fit=all_conditions["accuracy"]["memorability"],
                                           model=accuracy_model, model_type=accuracy_model_type)
        loadblindness_ana_results.fit_model(study, model=accuracy_model,
                                            conditions_to_fit=all_conditions["accuracy"]["loadblindness"],
                                            model_type=accuracy_model_type)
        workingmemory_ana_results.fit_model(study, model=accuracy_model,
                                            conditions_to_fit=all_conditions["accuracy"]["workingmemory"],
                                            model_type=accuracy_model_type)
    if RT_model:
        taskswitch_ana_results.fit_model(study, conditions_to_fit=all_conditions["RT"]["taskswitch"],
                                         model=PooledModelRTCostSimulations, model_type="pooled_model_RT")
        memorability_ana_results.fit_model(study, conditions_to_fit=all_conditions["RT"]["memorability"],
                                           model=PooledModelRTSimulations, model_type=RT_model_type)
        gonogo_ana_results.fit_model(study, conditions_to_fit=all_conditions["RT"]["gonogo"], model=RT_model,
                                     model_type=RT_model_type)


def init_from_traces_and_plot(study, accuracy_model=None, RT_model=None, accuracy_model_type=None, RT_model_type=None):
    if accuracy_model:
        gonogo_ana_results.run_visualisation(study, model=accuracy_model, model_type=accuracy_model_type,
                                             conditions_to_keep=all_conditions["accuracy"]["gonogo"])
        taskswitch_ana_results.run_visualisation(study, conditions_to_keep=all_conditions["accuracy"]["taskswitch"],
                                                 model=accuracy_model, model_type=accuracy_model_type)
        memorability_ana_results.run_visualisation(study, conditions_to_keep=all_conditions["accuracy"]["memorability"],
                                                   model=PooledModel, model_type=accuracy_model_type)
        enumeration_ana_results.run_visualisation(study=study, model=accuracy_model, model_type=accuracy_model_type,
                                                  conditions_to_keep=all_conditions["accuracy"]["enumeration"])
        moteval_ana_results.run_visualisation(study, model=accuracy_model, model_type=accuracy_model_type,
                                              conditions_to_keep=all_conditions["accuracy"]["moteval"])
        loadblindness_ana_results.run_visualisation(study, model=accuracy_model, model_type=accuracy_model_type,
                                                    conditions_to_keep=all_conditions["accuracy"]["loadblindness"])
        workingmemory_ana_results.run_visualisation(study, model=accuracy_model, model_type=accuracy_model_type,
                                                    conditions_to_keep=all_conditions["accuracy"]["workingmemory"])

    if RT_model:
        taskswitch_ana_results.run_visualisation(study, conditions_to_keep=all_conditions["RT"]["taskswitch"],
                                                 model=PooledModelRTCostSimulations, model_type=RT_model_type)
        memorability_ana_results.run_visualisation(study, conditions_to_keep=all_conditions["RT"]["memorability"],
                                                   model=RT_model, model_type=RT_model_type)
        gonogo_ana_results.run_visualisation(study, model=RT_model, model_type=RT_model_type,
                                             conditions_to_keep=all_conditions["RT"]["gonogo"])


def get_csv_for_all(study):
    enumeration_ana_results.fit_model(study=study, conditions_to_fit=all_conditions["accuracy"]["enumeration"],
                                      save_lfa=True)
    gonogo_ana_results.fit_model(study, conditions_to_fit=all_conditions["accuracy"]["gonogo"], save_lfa=True)
    taskswitch_ana_results.fit_model(study, conditions_to_fit=all_conditions["accuracy"]["taskswitch"], save_lfa=True)
    moteval_ana_results.fit_model(study, conditions_to_fit=all_conditions["accuracy"]["moteval"], save_lfa=True)
    # memorability_ana_results.fit_model(study, conditions_to_fit=all_conditions["accuracy"]["memorability"],
    #                                    save_lfa=True)
    loadblindness_ana_results.fit_model(study, conditions_to_fit=all_conditions["accuracy"]["loadblindness"],
                                        save_lfa=True)
    workingmemory_ana_results.fit_model(study, conditions_to_fit=all_conditions["accuracy"]["workingmemory"],
                                        save_lfa=True)
    # memorability_ana_results.fit_model(study, conditions_to_fit=all_conditions["RT"]["memorability"], save_lfa=True)


if __name__ == '__main__':
    get_csv_for_all(study)
    # run_all(study, RT_model=PooledModelRTSimulations, RT_model_type="pooled_model_RT")
    # run_all(study, accuracy_model=PooledModel, accuracy_model_type="pooled_model")
    # init_from_traces_and_plot(study, RT_model=PooledModelRTSimulations, RT_model_type="pooled_model_RT")
    # init_from_traces_and_plot(study, accuracy_model=PooledModel, accuracy_model_type="pooled_model")
