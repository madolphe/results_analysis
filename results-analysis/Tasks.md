# MOT 

### Task description:

### Raw CSV:
The MOT task proposes 54 trials:
nb_trials = 54 = (2 cdt of nb targets) * (3 cdts of speed) * (3trials) * (3blocks)

| participant_id | results_response | results_correct                              | results_speed_stim         | results_num_target         | results_ind_condition | results_rt        |
|----------------|------------------|----------------------------------------------|----------------------------|----------------------------|-----------------------|-------------------|
| 212            | 2,6,1,...        | 0.8, 0.1, 0.6...                             | 8,1,4,8...                 | 5,5,5,3...                 | 2,0,5,3               | 4334,1221,1112... |
| integer        | list of integer  | list of floats (size=54): accuracy per trial | list of integers (size=54) | list of integers (size=54) |                       |                   |

#### Useless : results_ind_cond

To randomize the sequence of conditions, several variables are used:
list_cond contains all possible pairs of conditions (e.g [[3,1], [3,4]... [5,8]]) [size = 6]
trial_stimcond contains a random order for proposing all possible pairs of conditions (e.g [0, 5, 2, 3...]) [size = 6]
results_ind_condition corresponds to trial_stimcond[index_trial] i.e corresponds to results_num_target,
results_speed_stim
This only allows to check that the random order is coherent.

#### Useless : results_responses; index of all buttons clicked ==> [size == 27*(3+5) = 216]

For a specific trial, objects names (target + distractors) are sorted so that we have first targets then distractors.
As an example, for a trial with 3 targets, the objects [0,1,2] will correspond to targets and [3,4,5,6,7,8,9] to
distractors.
Results_response records the name of the objects clicked; if the response is [1,0,6] it is possible to know that the
participant has cliked correctly on 2 targets (named 1 and 0) and failed on one distractor (named 6).
Results_response should then have a size of 216 --> 3 blocks * 3 trials per condition * 3 speed conditions * (
3responses + 5 response) = 216
results_response records the button_index clicked, for a trial with 3 targets, 3 values will represent the response
clicked, all values superior to nb_target are wrong (buttons)

#### Important:

results_speed_stim : speed of trial (len = 54) \
results_num_target : nb of targets (len = 54) \
results_correct : nb_targets_retrieved / nb_total_targets \
Number of answer slots = number of targets proposed ==> [F1-score = precision = recall]

### Data Processing:
For each trial, we have the accuracy (i.e nb_targets_retrieved / nb_targets). We compute the mean over the session
for each condition [speed=m, nb-targets=n] i.e m-speed-n-nb-target + we average for a specific
condition of speed and a specific condition of accuracy. \
In addition to mean accuracy, we also compute nb_trials correct which corresponds to number of trials that were
correctly retrieve (i.e all targets retrieved, acc = 1). This second binary measure is maybe less precise. 

### ATTENTION ==>  If use nb_corrects for MOT in bayesian t-test; maybe not the best (à tester avec une modélisation qui prend l'accuracy)

# UFOV

### Task description:

### Raw CSV:
UFOV task difficulty is controlled with 3up/1down staircase (max nb_trials = 72).
The difficulty parameter is the duration of the central stimulus proposed. It starts at 16 frames (i.e ).
Conditions?

| border_eccentricity_used                                                                | eccentricity_proposed      | direction_proposed               | central_target_proposed | difficulty_proposed                                                                    | measured_difficulties_duration | measured_difficulties_frame_count  | results                        | instruction_scene_duration | results_rt                     | experiment_duration |
|-----------------------------------------------------------------------------------------|----------------------------|----------------------------------|-------------------------|----------------------------------------------------------------------------------------|--------------------------------|------------------------------------|--------------------------------|----------------------------|--------------------------------|---------------------|
| maximum border of the circle used. This is dependant on the screen size of participants | useless as always the same | direction of peripheral stimulus | stimulus type (G or H)  | level of difficulty (start at 16 and range from 99 to 1 following staircase algorithm) | real stimulus presentation     | same but in nb of frames presented | true if both tasks are correct | random duration            | reaction time for central task | duration in ms      |

### Data Processing:
Average difficulty on the last 5 trials. The difficulty level goes from 99 (=n frames) to 1 (=16 frames). 
Nb_trials to complete the task (from 72 to 1). Nb_trials is quite interesting because of possible ceiling effect. 


# Enumeration

### Task description:
Conditions are number of dots presented. [5, 6, 7, 8, 9] 
Each condition is presented 20 times. 
Number of blocks ?

### Raw CSV:
results_responses corresponds to number of dots answered by participant. 
results_targetvalue corresponds to the target value. 
results_rt is the reaction time for answer.

### Data Processing:
result_correct is a list of result for each trial (1 if correct, 0 otherwise).
For each condition, we compute how many correct trials there are + the success rate i.e (nb_correct / nb_trials).


# GONOGO

### Task description:

### Raw CSV:
results_ind_previous 
results_response contains all response by participant with 0 no click expected, 1 if hit (clik when expected) and 2 if false alarm. 
results_targetvalue contains all targets, "3" 

### Data Processing:


# Task switch

### Task description:

### Raw CSV:

### Data Processing: